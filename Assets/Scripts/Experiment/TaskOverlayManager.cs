using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class TaskOverlayManager : MonoBehaviour
{
    private TextMeshProUGUI titleText;
    private TextMeshProUGUI instructionText;
    // Start is called before the first frame update
    private string originalTitleText;
    private string originalInstructionText;
    void Awake()
    {
        titleText = transform.Find("TitleTask").GetComponent<TextMeshProUGUI>();
        instructionText = transform.Find("Instructions").GetComponent<TextMeshProUGUI>();
        if (titleText == null || instructionText == null)
        {
            Debug.LogError("TitleTask or Instructions TextMeshProUGUI component not found in TaskOverlayManager.");
        }
        originalTitleText = titleText.text;
        originalInstructionText = instructionText.text;
    }
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void SetTaskNumber(int taskNumber)
    {
        titleText.text = originalTitleText.Replace("?", taskNumber.ToString());
    }

    public void setNBackLevel(int nBackLevel)
    {
        instructionText.text = originalInstructionText.Replace("?", nBackLevel.ToString());
    }
}
