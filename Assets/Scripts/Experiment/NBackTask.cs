using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using System.IO;
using Experiment;

public class NBackTask : MonoBehaviour
{

    public int DefaultNBack = 0;
    public float StimulusDelay = 2.25f;
    public uint TotalStimuli = 10;
    public string StimulusAudioFilePattern = "#_female";
    public bool IsPracticeMode = false;
    public bool saveResultsToFile = false;
    public AudioSource AudioSource;

    private int currentNBack;
    private List<PySenderData.NBackData> stimulusSequence = new List<PySenderData.NBackData>();
    private List<byte> practiceSequence = new List<byte>{3, 1, 4, 1, 1, 5, 5, 6, 5, 3};
    private int currentStimulusIndex = 0;
    private float lastStimulusTime;
    private long lastUserClickTime;
    private bool isTaskActive = false;
    private bool hasUserClicked = false;
    private AudioClip next_clip;
    private const string k_audioFolderPath = "Audio/NBackStimuli/";
    private const string k_resultsFolderPath = "Assets/Data/NBack/";
    // Start is called before the first frame update
    void Start()
    {
        currentNBack = DefaultNBack;
        lastUserClickTime = 0;

        // Ensure results directory exists if needed
        if (saveResultsToFile && !Directory.Exists(k_resultsFolderPath))
        {
            Directory.CreateDirectory(k_resultsFolderPath);
        }

        // Initialize shared memory
        PySender.Instance.InitializeNBackDataSharedMemory(TotalStimuli);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Tab) && isTaskActive)
        {
            hasUserClicked = true;
            lastUserClickTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        }
        if (Input.GetKeyDown(KeyCode.B) && !isTaskActive)
        {
            beginTask();
        }
        if (isTaskActive)
        {
            if (Time.time - lastStimulusTime >= StimulusDelay)
            {
                if (currentStimulusIndex > 0)
                {
                    checkResponse();
                }
                if (currentStimulusIndex < stimulusSequence.Count)
                {
                    PySenderData.NBackData currentStimulus = stimulusSequence[currentStimulusIndex];
                    prepareStimulus(currentStimulus.Stimulus);
                    AudioSource.Play();
                    currentStimulus.TimeStamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    stimulusSequence[currentStimulusIndex] = currentStimulus;
                    lastStimulusTime = Time.time;
                    currentStimulusIndex++;
                }
                else
                {
                    endTask();
                }
            }
        }
    }

    /// <summary>
    /// Sets the N value for the N-Back task (overwrites default).
    /// </summary>
    /// <param name="nBack">The N value to set.</param>
    public void setNBack(int nBack)
    {
        currentNBack = nBack;
    }

    void generateStimulusSequence()
    {
        stimulusSequence.Clear();
        if (IsPracticeMode)
        {
            // Use predefined practice sequence
            foreach (var stimulus in practiceSequence)
            {
                stimulusSequence.Add(new PySenderData.NBackData(stimulus, (byte)currentNBack));
            }
            return;
        }
        for (uint i = 0; i < TotalStimuli; i++)
        {
            // 0-9 audio numbers
            byte stimulus = (byte)UnityEngine.Random.Range(0, 10);
            stimulusSequence.Add(new PySenderData.NBackData(stimulus, (byte)currentNBack));
        }
    }

    void prepareStimulus(int stimulus)
    {
        string audioFileName = k_audioFolderPath + StimulusAudioFilePattern.Replace("#", stimulus.ToString());
        next_clip = Resources.Load<AudioClip>(audioFileName);
        AudioSource.clip = next_clip;
    }

    void checkResponse()
    {
        // Sanity check
        if (currentStimulusIndex <= 0)
        {
            Debug.LogError("checkResponse called when currentStimulusIndex <= 0");
            return;
        }
        int stimulusToMatchIndex = currentStimulusIndex - 1 - currentNBack;
        PySenderData.NBackData actualStimulus = stimulusSequence[currentStimulusIndex - 1];
        actualStimulus.ParticipantResponse = hasUserClicked ? (byte)1 : (byte)0;
        if (stimulusToMatchIndex >= 0)
        {
            PySenderData.NBackData expectedStimulus = stimulusSequence[stimulusToMatchIndex];
            if (hasUserClicked)
            {
                actualStimulus.ResponseTimeStamp = lastUserClickTime;
                actualStimulus.IsCorrect = (expectedStimulus.Stimulus == actualStimulus.Stimulus) ? (byte)1 : (byte)0;
            }
            else
            {
                actualStimulus.IsCorrect = (expectedStimulus.Stimulus != actualStimulus.Stimulus) ? (byte)1 : (byte)0;
            }
        } else
        {
            // No response expected for first N stimuli
            actualStimulus.IsCorrect = !hasUserClicked ? (byte)1 : (byte)0;
        }
        stimulusSequence[currentStimulusIndex - 1] = actualStimulus;
        hasUserClicked = false; // Reset for next stimulus
        PySender.Instance.UpdateNBackData(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(), actualStimulus, currentStimulusIndex - 1);
    }

    /// <summary>
    /// Begins the N-Back task.
    /// Generates a new random stimulus sequence.
    /// </summary>
    public void beginTask()
    {
        Debug.Log("Beginning " + currentNBack + "-Back Task");
        generateStimulusSequence();
        PySender.Instance.UpdateNBackData(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(), stimulusSequence.ToArray());
        isTaskActive = true;
        currentStimulusIndex = 0;
        lastStimulusTime = Time.time;
    }

    public void endTask()
    {
        writeResultsToFile();
        Debug.Log("Ending " + currentNBack + "-Back Task");
        Debug.Log("Score: " + (stimulusSequence.Count > 0 ? (float)stimulusSequence.Count(s => s.IsCorrect == 1) / stimulusSequence.Count * 100f : 0f) + "%");
        Debug.Log("Correct Responses: [" + string.Join(", ", stimulusSequence.Select((s, i) => s.IsCorrect == 1 ? "1" : "0").ToArray()) + "]");
        // Calculate user response delays
        List<long> responseDelays = new List<long>();
        foreach (var s in stimulusSequence)
        {
            if (s.ParticipantResponse == 1)
            {
                responseDelays.Add(s.ResponseTimeStamp - s.TimeStamp);
            }
            else {
                responseDelays.Add(-1); // Indicate no response
            }
        }
        Debug.Log("Response Delays: [" + string.Join(", ", responseDelays) + "] ms");
        isTaskActive = false;
    }

    /// <summary>
    /// Writes the results of the N-Back task to a CSV file.
    /// </summary>
    private void writeResultsToFile()
    {
        if (!saveResultsToFile) return;
        if (IsPracticeMode) return;

        string filename = k_resultsFolderPath + DateTime.Now.ToString("yyyyMMddHHmmss") + "_NBackResults.csv";
        using (StreamWriter writer = new StreamWriter(filename))
        {
            writer.WriteLine("NBackLevel,Stimulus,StimulusTimestamp,UserResponded,IsCorrect,ResponseTimestamp");
            foreach (var s in stimulusSequence)
            {
                writer.WriteLine($"{s.NBackLevel},{s.Stimulus},{s.TimeStamp},{s.ParticipantResponse},{s.IsCorrect},{s.ResponseTimeStamp}");
            }
        }
        Debug.Log("Results written to " + filename);
    }
}
