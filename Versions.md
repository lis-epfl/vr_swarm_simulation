# Last stable working version

PS C:\Users\sriram\vr_swarm_simulation\Assets\Scripts\swarm> py -0   
Installed Pythons found by C:\WINDOWS\py.exe Launcher for Windows
 -3.10-64 *

PS C:\Users\sriram\vr_swarm_simulation\Assets\Scripts\swarm> py -3.10 -m  pip list
Package           Version
----------------- ------------
asttokens         3.0.0
colorama          0.4.6
coloredlogs       15.0.1
comm              0.2.3
contourpy         1.3.2
cycler            0.12.1
debugpy           1.8.17
decorator         5.2.1
exceptiongroup    1.3.0
executing         2.2.1
filelock          3.19.1
flatbuffers       25.9.23
fonttools         4.60.1
fsspec            2025.9.0
humanfriendly     10.0
ipykernel         7.0.1
ipython           8.37.0
jedi              0.19.2
Jinja2            3.1.6
jupyter_client    8.6.3
jupyter_core      5.9.0
kiwisolver        1.4.9
MarkupSafe        2.1.5
matplotlib        3.10.7
matplotlib-inline 0.1.7
ml_dtypes         0.5.3
mpmath            1.3.0
nest-asyncio      1.6.0
networkx          3.3
numpy             2.1.2
onnx              1.19.1
onnxruntime       1.23.1
opencv-python     4.12.0.88
packaging         25.0
parso             0.8.5
pillow            11.3.0
pip               25.2
platformdirs      4.5.0
prompt_toolkit    3.0.52
protobuf          6.33.0
psutil            7.1.0
pure_eval         0.2.3
pycocotools       2.0.10
Pygments          2.19.2
pyparsing         3.2.5
pyreadline3       3.5.4
python-dateutil   2.9.0.post0
pyzmq             27.1.0
segment_anything  1.0
setuptools        65.5.0
six               1.17.0
stack-data        0.6.3
sympy             1.14.0
torch             2.9.0+cu126
torchvision       0.24.0+cu126
tornado           6.5.2
traitlets         5.14.3
typing_extensions 4.15.0
wcwidth           0.2.14


PS C:\Users\sriram\vr_swarm_simulation> nvidia-smi
Thu Oct 16 14:25:30 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090      WDDM  |   00000000:01:00.0  On |                  Off |
|  0%   41C    P8             25W /  450W |    1113MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1776    C+G   ...Editor\2022.3.62f2\Editor\Unity.exe      N/A      |
|    0   N/A  N/A      1884    C+G   ...s\System32\ApplicationFrameHost.exe      N/A      |
|    0   N/A  N/A      3964    C+G   ...Programs\Microsoft VS Code\Code.exe      N/A      |
|    0   N/A  N/A      5024    C+G   ...Editor\2022.3.62f2\Editor\Unity.exe      N/A      |
|    0   N/A  N/A      6904    C+G   C:\Windows\explorer.exe                     N/A      |
|    0   N/A  N/A      9532    C+G   C:\Windows\System32\ShellHost.exe           N/A      |
|    0   N/A  N/A     11112    C+G   ...tionsPlus\logioptionsplus_agent.exe      N/A      |
|    0   N/A  N/A     14080    C+G   ...cal\Microsoft\OneDrive\OneDrive.exe      N/A      |
|    0   N/A  N/A     14960    C+G   ...nt.CBS_cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A     14984    C+G   ...2txyewy\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A     15208    C+G   ...siveControlPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A     17148    C+G   ...on\141.0.3537.71\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     20288    C+G   ...US\ArmouryDevice\asus_framework.exe      N/A      |
|    0   N/A  N/A     23136    C+G   ...ogram Files\Unity Hub\Unity Hub.exe      N/A      |
|    0   N/A  N/A     25196    C+G   ...crosoft\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A     26128    C+G   ...Editor\2022.3.62f2\Editor\Unity.exe      N/A      |
|    0   N/A  N/A     26928    C+G   ...m Files\Mozilla\Firefox\firefox.exe      N/A      |
|    0   N/A  N/A     27620    C+G   ...__ppwjx1n5r4v9t\Blender\blender.exe      N/A      |
|    0   N/A  N/A     28816    C+G   ...m Files\Mozilla\Firefox\firefox.exe      N/A      |
|    0   N/A  N/A     29652    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe      N/A      |
+-----------------------------------------------------------------------------------------+
