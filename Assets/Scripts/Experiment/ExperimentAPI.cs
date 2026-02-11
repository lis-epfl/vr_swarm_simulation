using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

namespace Experiment
{
    public class API : MonoBehaviour
    {
        [Header("HTTP Server")]
        [SerializeField] private int port = 8080;
        [SerializeField] private string listenAddress = "http://localhost";

        [Header("References")]
        [SerializeField] private ExperimentFSM fsm;

        private HttpListener listener;
        private Thread listenerThread;
        private readonly ConcurrentQueue<Action> mainThreadActions = new ConcurrentQueue<Action>();
        private bool isRunning;

        [Serializable]
        private class StateRequest
        {
            public string state;
        }

        private struct ApiResult
        {
            public bool Ok;
            public string Error;

            public static ApiResult Success()
            {
                ApiResult result = new ApiResult
                {
                    Ok = true,
                    Error = string.Empty
                };
                return result;
            }

            public static ApiResult Fail(string error)
            {
                ApiResult result = new ApiResult
                {
                    Ok = false,
                    Error = error
                };
                return result;
            }
        }

        private void Awake()
        {
            if (fsm == null)
                fsm = FindObjectOfType<ExperimentFSM>();
        }

        private void Start()
        {
            StartServer();
        }

        private void Update()
        {
            Action action;
            while (mainThreadActions.TryDequeue(out action))
            {
                try
                {
                    action?.Invoke();
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Experiment API main-thread action failed: {ex}");
                }
            }
        }

        private void OnDestroy()
        {
            StopServer();
        }

        private void StartServer()
        {
            if (isRunning)
                return;

            if (string.IsNullOrWhiteSpace(listenAddress))
                listenAddress = "http://localhost";

            string prefix = listenAddress.TrimEnd('/') + ":" + port + "/";
            listener = new HttpListener();
            listener.Prefixes.Add(prefix);

            try
            {
                listener.Start();
            }
            catch (HttpListenerException ex)
            {
                Debug.LogError($"Experiment API failed to start on {prefix}. Error: {ex.Message}");
                return;
            }

            isRunning = true;
            listenerThread = new Thread(ListenLoop)
            {
                IsBackground = true,
                Name = "ExperimentAPIListener"
            };
            listenerThread.Start();

            Debug.Log($"Experiment API listening on {prefix}");
        }

        private void StopServer()
        {
            if (!isRunning)
                return;

            isRunning = false;

            try
            {
                listener?.Stop();
                listener?.Close();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Experiment API error while stopping: {ex}");
            }

            listener = null;
            listenerThread = null;
        }

        private void ListenLoop()
        {
            while (isRunning && listener != null && listener.IsListening)
            {
                HttpListenerContext context = null;
                try
                {
                    context = listener.GetContext();
                }
                catch (HttpListenerException)
                {
                    break;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }

                if (context == null)
                    continue;

                try
                {
                    HandleRequest(context);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Experiment API request handling failed: {ex}");
                    WriteJsonResponse(context.Response, 500, new { error = "internal_error" });
                }
            }
        }

        private void HandleRequest(HttpListenerContext context)
        {
            HttpListenerRequest request = context.Request;
            HttpListenerResponse response = context.Response;

            response.Headers["Access-Control-Allow-Origin"] = "*";
            response.Headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
            response.Headers["Access-Control-Allow-Headers"] = "Content-Type";

            if (request.HttpMethod == "OPTIONS")
            {
                response.StatusCode = 204;
                response.Close();
                return;
            }

            string path = request.Url.AbsolutePath.TrimEnd('/').ToLowerInvariant();

            if (request.HttpMethod == "GET" && path == "/api/state")
            {
                ExperimentFSM.ExperimentStateSnapshot snapshot = ExecuteOnMainThread(() => fsm != null ? fsm.GetStateSnapshot() : null, 1000);
                if (snapshot == null)
                {
                    WriteJsonResponse(response, 503, new { error = "fsm_unavailable" });
                    return;
                }

                string json = JsonUtility.ToJson(snapshot);
                WriteJsonResponse(response, 200, json, rawJson: true);
                return;
            }

            if (request.HttpMethod == "GET" && path == "/api/operatorclicked")
            {
                bool ok = ExecuteOnMainThread(() =>
                {
                    if (fsm == null) return false;
                    fsm.NotifyOperatorClicked();
                    return true;
                }, 1000);

                if (!ok)
                {
                    WriteJsonResponse(response, 503, new { error = "fsm_unavailable" });
                    return;
                }

                WriteJsonResponse(response, 200, new { ok = true });
                return;
            }

            if (request.HttpMethod == "POST" && path == "/api/state")
            {
                string body;
                using (StreamReader reader = new StreamReader(request.InputStream, request.ContentEncoding))
                {
                    body = reader.ReadToEnd();
                }

                StateRequest stateRequest = null;
                try
                {
                    stateRequest = JsonUtility.FromJson<StateRequest>(body);
                }
                catch
                {
                    stateRequest = null;
                }

                if (stateRequest == null || string.IsNullOrWhiteSpace(stateRequest.state))
                {
                    WriteJsonResponse(response, 400, new { error = "missing_state" });
                    return;
                }

                ApiResult result = ExecuteOnMainThread(() =>
                {
                    if (fsm == null)
                        return ApiResult.Fail("fsm_unavailable");

                    if (!fsm.RequestTransitionTo(stateRequest.state, out string transitionError))
                        return ApiResult.Fail(transitionError);

                    return ApiResult.Success();
                }, 1000);

                if (!result.Ok)
                {
                    WriteJsonResponse(response, 400, new { error = result.Error });
                    return;
                }

                WriteJsonResponse(response, 200, new { ok = true });
                return;
            }

            WriteJsonResponse(response, 404, new { error = "not_found" });
        }

        private T ExecuteOnMainThread<T>(Func<T> func, int timeoutMs)
        {
            if (!isRunning)
                return default;

            ManualResetEventSlim done = new ManualResetEventSlim(false);
            T result = default;

            mainThreadActions.Enqueue(() =>
            {
                try
                {
                    result = func();
                }
                finally
                {
                    done.Set();
                }
            });

            if (!done.Wait(timeoutMs))
            {
                return default;
            }

            return result;
        }

        private void WriteJsonResponse(HttpListenerResponse response, int statusCode, object payload, bool rawJson = false)
        {
            response.StatusCode = statusCode;
            response.ContentType = "application/json";

            string json;
            if (rawJson)
            {
                json = payload as string ?? "{}";
            }
            else
            {
                json = JsonUtility.ToJson(payload);
            }

            byte[] buffer = Encoding.UTF8.GetBytes(json);
            response.ContentLength64 = buffer.Length;
            using (Stream output = response.OutputStream)
            {
                output.Write(buffer, 0, buffer.Length);
            }
        }
    }
}
