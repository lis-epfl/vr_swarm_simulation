using System;
using UnityEngine;

namespace Experiment
{
    public abstract class ExperimentFSMBase : MonoBehaviour
    {
        [Serializable]
        public class ExperimentStateSnapshot
        {
            public long timestamp => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            public string state;
            public string previousState;
            public string nextState;
            public int currentTask;
            public int currentTrial;
            public int totalTaskNumber;
            public int totalTrialNumber;
            public long stateEnterTimestamp;
        }

        /// <summary>
        /// Returns the current state of the experiment FSM as a serializable snapshot.
        /// </summary>
        public abstract ExperimentStateSnapshot GetStateSnapshot();

        /// <summary>
        /// Returns an array of all available state names in this FSM.
        /// </summary>
        public abstract string[] GetAvailableStates();

        /// <summary>
        /// Notifies the FSM that an external operator clicked (triggered via HTTP API).
        /// </summary>
        public abstract void NotifyOperatorClicked();

        /// <summary>
        /// Requests a transition to a named state.
        /// </summary>
        /// <param name="stateName">Name of the target state (case-insensitive).</param>
        /// <param name="error">Output error message if transition fails.</param>
        /// <returns>True if transition was successful, false otherwise.</returns>
        public abstract bool RequestTransitionTo(string stateName, out string error);
    }
}
