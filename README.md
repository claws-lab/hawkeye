## HawkEye: A Robust Reputation System for Community-based Counter-Misinformation

#### Authors : [Rohit Mujumdar](https://rohitmujumdar.github.io/), [Srijan Kumar](http://cs.stanford.edu/~srijan)

<!--#### [Link to the paper]()
#### [Link to the slides]()
#### [Brief video explanation]()-->

### About HawkEye

Identifying misinformation is a critical task on web and social media platforms. Recent efforts have focused on leveraging the community of ordinary users to detect, counter, and curb misinformation. Twitter launched a community-driven misinformation detection service called [Birdwatch](https://blog.twitter.com/en_us/topics/product/2021/introducing-birdwatch-a-community-based-approach-to-misinformation), where users provide notes to label tweets as misleading or not, and rate other users' notes as being 'helpful' or not. However, malicious users can inject fake notes and helpfulness ratings to manipulate the system for their gains. In this work, we investigate the robustness of Birdwatch against adversaries. We show that the current Birdwatch system is vulnerable to adversarial attacks - using only a few fake accounts, an adversary can promote any random note as one of the top ranking notes. 

To overcome this vulnerability, we propose **HawkEye**, a graph-based recursive algorithm that leverages the global graph structure to quantifyall the quality metrics. Since many users will only write andrate a few notes and many tweets will only have a few notes, we introduce a Laplacian smoothing technique to overcomethis cold-start problem. We posit that HawkEye will be more robust to adversaries.

We compare the Birdwatch and HawkEye models' robustness against an attacker whose goal is to manipulate the ranking of notes. We show that our proposed HawkEye algorithm is more robust against this attack. Furthermore, we show that the HawkEye algorithm performs better than the Birdwatch system in identifying accurate and misleading tweets in both unsupervised and supervised settings. 

If you make use of this code, the HawkEye algorithm, please cite the following paper:
```
 @inproceedings{mujumdar2021hawkeye,
  title={HawkEye: A Robust Reputation System for Community-based Counter-Misinformation},
  author={Mujumdar, Rohit and Kumar, Srijan},
  booktitle={2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2021)},
  year={2021},
  organization={IEEE}
}
```

<!--### Short Video Explanation of HawkEye (External Link to YouTube)

[![HawkEye short video]()]()-->

### Repository Structure

All code and data is stored in `src/code/` and `src/data/` folder respectively. 

### Dataset
Links to datasets used in the paper:
- [Birdwatch Data](https://twitter.github.io/birdwatch/contributing/download-data/)


### Dataset format

The data files used for this work are stored under the `data/` folder, one CSV file each for notes and ratings. The filename is according to the nomenclature on the official [Birdwatch data download page](https://twitter.github.io/birdwatch/contributing/download-data/). The format of CSV files and the meaning of the columns can be found on the [Birdwatch data download page](https://twitter.github.io/birdwatch/contributing/download-data/). 


### Code setup and Requirements

The code in this repo uses Python 3.7 and recent versions of Pandas, numpy, sklearn, tqdm, matplotlib, seaborn, and tweepy. `requirements.txt` can be found in the `code/` folder. You can install all the required packages using the following command:
```
    $ pip install -r requirements.txt
```

A `results/` directory needs to be created in the `code/` folder.


### Running the HawkEye scripts

Jupyter notebooks : Run all the cells in chrological order. <br>
Python scripts : Run all scripts using `python file-name.py` <br>
Comments in the code description of the goal of the script and the functions used in it. Instructions about what to change in the code are also present.  
