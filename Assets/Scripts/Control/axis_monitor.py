"""Quick diagnostic: dump every joystick axis live + show the panorama toggle.

Run this INSTEAD of readController.py while you press the click switch, to confirm:
  1. Which axis actually moves when you press the switch (should be axis 5).
  2. That it rests at ~-1 and jumps to ~+1 while held.
  3. That the rising-edge toggle flips panorama_enabled on each press.

    conda activate stitching   # or any env with pygame
    python axis_monitor.py

Ctrl+C to quit.
"""
import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick detected.")
    raise SystemExit(1)

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick: {joystick.get_name()}  |  {joystick.get_numaxes()} axes")

panorama_enabled = True
prev_click = -1.0

try:
    while True:
        pygame.event.pump()

        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]

        click = joystick.get_axis(5)
        if click > 0.5 and prev_click <= 0.5:
            panorama_enabled = not panorama_enabled
            print(f"\n>>> CLICK detected -> panorama_enabled = {panorama_enabled}")
        prev_click = click

        axes_str = "  ".join(f"a{i}:{v:+.2f}" for i, v in enumerate(axes))
        print(f"\r{axes_str}   pano={panorama_enabled}   ", end="", flush=True)

        time.sleep(0.05)
except KeyboardInterrupt:
    pygame.quit()
