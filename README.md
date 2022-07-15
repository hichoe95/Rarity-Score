# Rarity Score

[Paper: Rarity Score : A New Metric to Evaluate the Uncommonness of Synthesized Images](https://arxiv.org/abs/2206.08549)

Jiyeon Han<sup>1</sup>, Hwanil Choi<sup>1</sup>, Yunjey Choi<sup>2</sup>, 
Junho Kim<sup>2</sup>, Jungwoo Ha<sup>2</sup>, Jaesik Choi<sup>1</sup>    


<sup>1</sup> <sub>Graduate School of AI, KAIST</sub>
<sup>2</sup> <sub>AI Lab, NAVER Corp.</sub>

Evaluation metrics in image synthesis play a key role to measure performances of generative models. However, most metrics mainly focus on image fidelity. Existing diversity metrics are derived by comparing distributions, and thus they cannot quantify the diversity or rarity degree of each generated image. In this work, we propose a new evaluation metric, called `rarity score', to measure the individual rarity of each image synthesized by generative models. We first show empirical observation that common samples are close to each other and rare samples are far from each other in nearest-neighbor distances of feature space. We then use our metric to demonstrate that the extent to which different generative models produce rare images can be effectively compared. We also propose a method to compare rarities between datasets that share the same concept such as CelebA-HQ and FFHQ. Finally, we analyze the use of metrics in different designs of feature spaces to better understand the relationship between feature spaces and resulting sparse images. Code will be publicly available online for the research community.

<p align="center">
    <img src=sample.png width="900"> 
</p>

### Example
```python
from rarity_score import *
import numpy as np

feature_dim = 2048
num_samples = 100
nearest_k = 3

real_features = np.random.normal(size=(num_samples, feature_dim))
fake_features = np.random.normal(size=(num_samples, feature_dim))


manifold = MANIFOLD(real_features=real_features, fake_features=fake_features)

score, score_index = manifold.rarity(k=nearest_k)

print(score[score_index])
```



### License
```
Copyright 2022-present NAVER Corp.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
