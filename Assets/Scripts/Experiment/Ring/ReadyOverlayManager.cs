using UnityEngine;
using TMPro;

public class ReadyOverlayManager : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI trialNumberText;

    private void OnValidate()
    {
        // Auto-find TextMeshProUGUI if not assigned
        if (trialNumberText == null)
            trialNumberText = GetComponent<TextMeshProUGUI>();
    }

    /// <summary>
    /// Updates the trial number display on the ready screen
    /// </summary>
    public void SetTrialNumber(int currentTrial, int totalTrials)
    {
        if (trialNumberText != null)
        {
            trialNumberText.text = $"Trial {currentTrial} of {totalTrials}";
        }
        else
        {
            Debug.LogWarning("[ReadyOverlayManager] TextMeshProUGUI component not found");
        }
    }
}
