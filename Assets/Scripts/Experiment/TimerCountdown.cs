using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Experiment
{
    public class TimerCountdown : MonoBehaviour
    {
        [SerializeField]
        public float Minutes
        {
            get { return minutes; }
            set
            {
                if (value < 0)
                    minutes = 0;
                else
                    minutes = value;
            }
        }
        [SerializeField]
        public float Seconds
        {
            get { return seconds; }
            set
            {
                if (value < 0)
                    seconds = 0;
                else if (value >= 60)
                    seconds = 59;
                else
                    seconds = value;
            }
        }

        private float seconds = 0f;
        private float minutes = 0f;

        public TextMeshProUGUI countdownText;
        private float currentTime;

        public event System.Action OnCountdownFinished;
        // Start is called before the first frame update
        void Start()
        {
            // BeginCountdown();
        }

        // Update is called once per frame
        void Update()
        {
        }

        public void BeginCountdown()
        {
            currentTime = Minutes * 60 + Seconds;
            StartCoroutine(CountdownCoroutine());
        }

        public void StopCountdown()
        {
            StopAllCoroutines();
            countdownText.text = "0:00";
        }

        private IEnumerator CountdownCoroutine()
        {
            while (currentTime > 0)
            {
                // Format time as M:SS
                int minutes = Mathf.FloorToInt(currentTime / 60);
                int seconds = Mathf.FloorToInt(currentTime % 60);
                countdownText.text = string.Format("{0}:{1:00}", minutes, seconds);
                yield return new WaitForSeconds(1f);
                currentTime -= 1f;
            }
            countdownText.text = "0:00";
            OnCountdownFinished?.Invoke();
        }
    }
}
