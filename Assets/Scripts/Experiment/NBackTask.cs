using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;

public class NBackTask : MonoBehaviour
{

    public int DefaultNBack = 0;
    public float StimulusDelay = 2.25f;
    public int TotalStimuli = 10;
    public string StimulusAudioFilePattern = "#_female";
    public AudioSource AudioSource;
    private int currentNBack;
    private List<int> stimulusSequence = new List<int>();
    private int currentStimulusIndex = 0;
    private List<int> correctResponses = new List<int>();
    private float lastStimulusTime;
    private bool isTaskActive = false;
    private bool hasUserClicked = false;
    private AudioClip next_clip;
    private const string k_audioFolderPath = "Audio/NBackStimuli/";
    // Start is called before the first frame update
    void Start()
    {
        currentNBack = DefaultNBack;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Tab) && isTaskActive)
        {
            hasUserClicked = true;
        }
        if (Input.GetKeyDown(KeyCode.B) && !isTaskActive)
        {
            beginTask();
        }
        if (isTaskActive)
        {
            if (Time.time - lastStimulusTime >= StimulusDelay)
            {
                if (currentStimulusIndex < stimulusSequence.Count)
                {
                    if (currentStimulusIndex > 0)
                    {
                        checkResponse();
                    }
                    prepareStimulus(stimulusSequence[currentStimulusIndex]);
                    AudioSource.Play();
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

    public void setNBack(int nBack)
    {
        currentNBack = nBack;
    }

    void generateStimulusSequence()
    {
        stimulusSequence.Clear();
        stimulusSequence = new List<int>{3, 1, 4, 1, 5, 5, 5, 6, 5, 3}; // Predefined sequence for testing
        // for (int i = 0; i < TotalStimuli; i++)
        // {
        //     // 0-9 audio numbers
        //     int stimulus = Random.Range(0, 10);
        //     stimulusSequence.Add(stimulus);
        // }
    }

    void prepareStimulus(int stimuli)
    {
        string audioFileName = k_audioFolderPath + StimulusAudioFilePattern.Replace("#", stimuli.ToString());
        next_clip = Resources.Load<AudioClip>(audioFileName);
        AudioSource.clip = next_clip;
    }

    void checkResponse()
    {
        int stimulusToMatchIndex = currentStimulusIndex - 1 - currentNBack;
        if (stimulusToMatchIndex >= 0)
        {
            int expectedStimulus = stimulusSequence[stimulusToMatchIndex];
            int actualStimulus = stimulusSequence[currentStimulusIndex - 1];
            if (hasUserClicked)
            {
                if (expectedStimulus == actualStimulus)
                {
                    correctResponses.Add(1); // Correct
                    }
                else
                {
                    correctResponses.Add(0); // Incorrect
                }
            }
            else
            {
                if (expectedStimulus == actualStimulus)
                {
                    correctResponses.Add(0); // Missed
                }
                else
                {
                    correctResponses.Add(1); // Correct Rejection
                }
            }
        } else
        {
            // No response expected for first N stimuli
            correctResponses.Add(hasUserClicked ? 0 : 1); 
        }
        hasUserClicked = false; // Reset for next stimulus
    }

    public void beginTask()
    {
        Debug.Log("Beginning " + currentNBack + "-Back Task");
        generateStimulusSequence();
        correctResponses.Clear();
        isTaskActive = true;
        currentStimulusIndex = 0;
        lastStimulusTime = Time.time;
    }

    public void endTask()
    {
        Debug.Log("Ending " + currentNBack + "-Back Task");
        Debug.Log("Correct Responses: " + string.Join(", ", correctResponses));
        Debug.Log("Score: " + (correctResponses.Count > 0 ? (float)correctResponses.Sum() / correctResponses.Count * 100f : 0f) + "%");
        isTaskActive = false;
    }
}
