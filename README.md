# Meta evaluation of style transfer metrics

This GitHub repository is associated with the paper "Mind the Style Gap: Meta-Evaluation of Style and Attribute Transfer
Metrics" (https://arxiv.org/abs/2502.15022).
### Constructed test set
The paper presents a "new constructed test set" which can be found under the folder "Data". Or on  Hugging Face https://huggingface.co/datasets/APauli/style_eval_content_test

The paper presents a "constructed test set" which can evaluate metrics on 'content preservation' under large style shifts. We show that metrics/approaches for evaluating 'content preservation' on style transfer **must be style-aware**. Futher, similarity-based  metrics that do not condition on the style shift are widely used but conceptually unsuitable for the task of style transfer. Prior meta-evaluation studies have deemed similarity-based metrics a good fit; however, this is due to misleading results, which we demonstrate stem from the use of skewed data in the evaluation.

More details in the paper.

### LogProp
In addition, the paper presents a new efficient method, **LogProp**, on evaluating content preservation and style strength in style transfer, which utilises a small-sized LLM, e.g. 3B parameters.  LogProb uses likelihood estimates of the sentences-to-evaluate under different prior contexts. Different "instructions" are part of the context, namely both with and without mentioning the target style. Thereby making the evaluation method style-aware. Details in paper.


Example of test data:
``` python
origional_text ="I don't appreciate your lack of effort."
rewrite_text1 ="I would like to express my concern regarding the effort being put forth."
rewrite_text2 = "I would appreciate less effort in this area."
target_style = 'polite'
```
Example of using LogProp: 
```python
from LogProp import LogProp
method = LogProp()

method.predict(rewrite_text1,origional_text,target_style)
```

### benchmark
We provide the scripts for running different metrics for "content preservation", and some for evaluating "style strengh". Note that different scripts have different installation requirements. 

The script running the LogProp is named "run_logprob_on_data.py".

The paper presents benchmark results using previously human-annotated datasets on system output or references. We provide references to obtain this data, which should then be saved in the 'data' folder. 



 



### Data from other sources
from other sources:
- [Mir] (Mir84) is on the Yelp sentiment task and is annotated by 3 workers, where the mean ratings are released. Data is downloaded from  \url{github.com/passeul/style-transfer-model-evaluation}.  Mir et al. (2019).

- [Lai] (Lai3) supplied human annotations on system output on formality task, using a continuous scale from 1-100. Download at \url{https://github.com/laihuiyuan/eval-formality-transfer} with MIT License. Lai et al. (2022).

- [Scialom + Alva-m.] (ScialomD21) is on human ratings for human written output for a simplification sentence task. Download using the URL in the paper. No license specified. We have filtrated this data to obtain annotation in all three dimensions for the same data input (we check for an exact match on source sentence, rewrite, sentenceID), and we ended up with 65 samples annotated by 25 workers. 
[Alva-m.] is system output on a simplification task. Data form link above. We filter the data such that we have 135 samples with 11 annotations in all three dimensions because we favour more samples over the number of annotations per sample. Alva-Manchego et al. (2020) and Scialom et al. (2021b).

- [Ziegen] (ZeigenB11) is on rewriting inappropriate arguments to appropriated, download available at \url{https://github.com/timonziegenbein/inappropriateness-mitigation}. Ziegenbein et al. (2024).

- [Cao] (Cao6c) human evaluation on a task between transferring different styles of expertise in the medical domain. The authors have kindly shared the data with human ratings. Cao et al. (2020).



### Scripts to run Metrics on datasets

**<ins>Style conditional metrics:</ins>**

- **Our proposed** We experiment with the backbone models META-
LLAMA/LLAMA-3.2-3B-INSTRUCT and META-
LLAMA/LLAMA-3.1-8B-INSTRUCT downloaded
from https://huggingface.co/meta-llama.

- **Promting and LLM for evaluation score** 

**<ins> Lexical similarity:</ins>** 

- **BLEU** (Papineni et al., 2002) we use the python
package NLTK implementations of BLEU with
default settings.

- **Meteor** (Banerjee and Lavie, 2005) we use the
python package from Huggingface evaluate with
default settings.

**<ins>Semantic similarity:</ins>** 

- **BertScore** (Zhang et al., 2019) We use
the implementation from https://github.com/
Tiiiger/bert_score with the current recom-
mended backbone model MICROSOFT/DEBERTA-
XLARGE-MNLI.

- **BLEURT** (Sellam et al., 2020) we use
the python implemention from https:
//huggingface.co/Elron/bleurt-large-512
using Huggingface Transformer libary with the
backbone model ELRON/BLEURT-LARGE-512.

- **Cosine similarity embeddings** we use the Sen-
tenceTransformer library with Labse embeddings
SENTENCE-TRANSFORMERS/LABSE, (Feng et al.,
2022).

**<ins> Fact-based:</ins>** 
- **QuestEval** We use the implementations
from https://github.com/ThomasScialom/
QuestEval.

### outputs
link to onedrive file


### Installation
Python 3.12 and transformers 4.43.3 for running  

### cite


### References
- Remi Mir, Bjarke Felbo, Nick Obradovich, and Iyad
Rahwan. 2019. Evaluating style transfer for text
- Huiyuan Lai, Jiali Mao, Antonio Toral, and Malvina
Nissim. 2022. Human judgement as a compass to
navigate automatic metrics for formality transfer
- Thomas Scialom, Louis Martin, Jacopo Staiano,
Eric Villemonte de La Clergerie, and Benoît Sagot.
2021b. Rethinking automatic evaluation in sentence
simplification
- Fernando Alva-Manchego, Louis Martin, Antoine Bor-
des, Carolina Scarton, Benoît Sagot, and Lucia Spe-
cia. 2020. ASSET: A dataset for tuning and evalua-
tion of sentence simplification models with multiple
rewriting transformations.
_ Yixin Cao, Ruihao Shui, Liangming Pan, Min-Yen Kan,
Zhiyuan Liu, and Tat-Seng Chua. 2020. Expertise
style transfer: A new task towards better communi-
cation between experts and laymen.
- 
