# StartTrek

## Setup

This project uses `LunarLander-v3`, so `gymnasium` must be installed with the `box2d` extra.

### Recommended Python version

Python 3.12 is the smoothest option for this repository. Python 3.14 can work, but the `box2d-py` and `pygame` dependencies may need native build tools and system headers.

### Linux system packages for Python 3.14

If `pip install -r requirements.txt` fails while building `box2d-py` or `pygame`, install the following packages first:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev swig libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libjpeg-dev libfreetype6-dev
```

Then rerun:

```bash
pip install -r requirements.txt
```
