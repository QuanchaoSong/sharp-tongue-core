# sharp-tongue-core

### How to Run

#### 1. Install dependencies
```bash
git clone https://github.com/QuanchaoSong/sharp-tongue-core.git
cd sharp-tongue-core/

# it's better use "conda", but not necessary. 
# python can be other versions, not only 3.8
conda create -n thesis-test python=3.8
conda activate thesis-test

# install dependencies
pip install flask flask_cors
pip install openai replicate
pip install tenacity backoff
pip install nltk
pip install transformers
pip install torch
pip install pillow
pip install opencv-python
pip install deepface

# for clip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

#### 2 Additional operations for `nltk`
Open a new Terminal and start a Python runtime environment:

```bash
# in python
-> import nltk
-> nltk.download()
```
Just like this:

<img width="710" alt="Screen Shot 2023-06-06 at 3 45 28 PM" src="https://github.com/QuanchaoSong/sharp-tongue-core/assets/47345588/ec04b56a-479d-408a-8520-601523a93c8d">

Now a window like the following will pop up:

<img width="949" alt="Screen Shot 2023-06-06 at 3 36 48 PM" src="https://github.com/QuanchaoSong/sharp-tongue-core/assets/47345588/9b86629d-cde0-46b7-b074-70d1536c9492">

Make sure the "*Download Directory*" be as one of the followings (according to your computer directory):
```bash
- '/Users/albus/nltk_data'
- '/Users/albus/mambaforge/envs/thesis-test/nltk_data'
- '/Users/albus/mambaforge/envs/thesis-test/share/nltk_data'
- '/Users/albus/mambaforge/envs/thesis-test/lib/nltk_data'
- '/usr/share/nltk_data'
- '/usr/local/share/nltk_data'
- '/usr/lib/nltk_data'
- '/usr/local/lib/nltk_data'
```
Then double click the "*Popular packages*", wait patiently until the downloading is finished, and close the window. Quit the python runtime environment and close this Terminal.


#### 3 Run the program
For the backend server programs, there are actually 3 methods:

<img width="433" alt="Screen Shot 2023-06-06 at 6 14 50 PM" src="https://github.com/QuanchaoSong/sharp-tongue-core/assets/47345588/e024dd75-4ee8-4d00-9e80-2de98b12060e">

The fastest methods to get the result is `method-replicate` and `method-vit`, because the anlysis for the image context is moved to the *Replicate* cloud computers which is super fast. To test on these two methods, simply run:

```bash
python method-vit/server.py
# or
python method-replicate/server.py
```

For the last methods, it is the same to run it:

```bash
python method-blip2/server.py
```

*It's recommended to use the `method-replicate` or `method-vit` because it saves time.*
