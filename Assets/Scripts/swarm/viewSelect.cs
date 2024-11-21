using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class viewSelect : MonoBehaviour
{
    public GameObject Arena;
    public GameObject ArenaStitch;
    public enum ViewingStrategy
    {
        Default,
        Stitch
    }
    public ViewingStrategy viewingStrategy;

    private OVRPlayerController playerController;

    void Start()
    {
        playerController = FindObjectOfType<OVRPlayerController>();
        UpdatePlayerPosition();
    }

    void OnValidate()
    {
        UpdatePlayerPosition();
    }

    private void UpdatePlayerPosition()
    {
        if (playerController == null) return;

        switch (viewingStrategy)
        {
            case ViewingStrategy.Default:
                playerController.transform.position = Arena.transform.position;
                break;
            case ViewingStrategy.Stitch:
                playerController.transform.position = ArenaStitch.transform.position;
                break;
        }
    }
}
