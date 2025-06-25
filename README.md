# FuncVul
## FuncVul: An Effective Function Level Vulnerability Detection Model using LLM and Code Chunk

Software supply chain vulnerabilities arise when attackers exploit weaknesses by injecting vulnerable code into widely used packages or libraries within software repositories. While most existing approaches focus on identifying vulnerable packages or libraries, they often overlook the specific functions responsible for these vulnerabilities. Pinpointing vulnerable functions within packages or libraries is critical, as it can significantly reduce the risks associated with using open-source software. Identifying vulnerable patches is challenging because developers often submit code changes that are unrelated to vulnerability fixes. To address this issue, this paper introduces FuncVul, an innovative code chunk-based model for function-level vulnerability detection in C/C++ and Python, designed to identify multiple vulnerabilities within a function by focusing on smaller, critical code segments. To assess the model's effectiveness, we construct six code and generic code chunk based datasets using two approaches: (1) integrating patch information with large language models to label vulnerable samples and (2) leveraging large language models alone to detect vulnerabilities in function-level code. To design FuncVul vulnerability model, we utilise GraphCodeBERT fine tune model that captures both the syntactic and semantic aspects of code. Experimental results show that FuncVul outperforms existing state-of-the-art models, achieving an average accuracy of 87-92% and an F1 score of 86-92% across all datasets. Furthermore, we have demonstrated that our code-chunk-based FuncVul model improves 53.9% accuracy and 42.0% F1-score than the full function-based vulnerability prediction.  

Please cite the following paper to use this code and dataset in your research work.  


Sajal Halder, Muhammad Ejaz Ahmed, and Seyit Camtepe. FuncVul: An Effective Function Level Vulnerability Detection Model using LLM and Code Chunk. Accepted in 30th European Symposium on Research in Computer Security (ESORICS) 2025. Here is the arXiv paper link : http://arxiv.org/abs/2506.19453

In this research work, we aim to answer the following research questions. 
  
      (i) What modeling strategies can be employed to accurately detect function-level vulnerabilities?
      (ii) Does leveraging code chunks enhance model performance compared to analyzing full-function code?
      (iii) Does the FuncVul model leverage generalized code properties for vulnerability detection?
      (iv) How effective is our approach at detecting vulnerabilities in unseen projects?
      (v) How does the performance of our approach vary with different numbers of source lines in a code chunk?
      (vi) Is our proposed model capable of detecting multiple vulnerabilities within a single function’s code?
    


# Datasets Description 

Details about datasets 

| Dataset | Prompt Code Type               | Vulnerable Defined By   | Vulnerable   | Non-vulnerable |
|---------|--------------------------------|-------------------------|--------------|----------------|
| 1       | Code + Description Code Chunk  | LLM + Path Information  | 1810 (43.4%) | 2357 (56.6%)   | 
| 2       | Code Code Chunk                | LLM + Path Information  | 2120 (42.6%) | 2851 (57.4%)|
|3       | Code + Description Generic Code Chunk | LLM + Path Information | 1810 (43.4%)   | 2357 (56.6%)|
| 4       | Code Generic Code Chunk |    LLM + Path Information            | 2120 (42.6%)   | 2851 (57.4%)|
| 5       | Code + Description Code Chunk   | LLM                     | 3169 (50%)   | 3169 (50%) |
| 6       | Code Code Chunk                 | LLM                     | 6041 (50%)   |6041 (50%) |


### Dataset 1 and 2 contain the following columns:

  'cve': cve number
  
  'code_chunks': 3-line based code chunks based on LLM and Patch Information
  
  'vul_category': Type of velnerability (do not use in experiment analysis)
  
  'label': Vulnerability classification label. vulnerable (1) or non-vulnerable (0)

### Dataset 3 and 4 contain the following columns:

  'cve': cve number
  
  'generic_code_chunks': generic code convert from 3-line based code chunks
  
  'vul_category': Type of velnerability (do not use in experiment analysis)
  
  'label': Vulnerability classification label. vulnerable (1) or non-vulnerable (0)

### Dataset 5 and 6 contain the following columns:

  'cve': cve number
  
  'code_chunks': 3-line based code chunks based on LLM detection. 
  
  'label': Vulnerability classification label. vulnerable (1) or non-vulnerable (0)

#### For RQ5 analysis, we used different added lines (1,3,5,7,9, 10,15,20,25) based on code chunks for dataset 6, which are as follows. 

Dataset_6_line_1.json : 1-Line Extended-Based Code Chunk 

Dataset_6.json : 3-Line Extended-Based Code Chunk 

Dataset_6_line_5.json : 5-Line Extended-Based Code Chunk 

Dataset_6_line_7.json : 7-Line Extended-Based Code Chunk 

Dataset_6_line_9.json : 9-Line Extended-Based Code Chunk 

Dataset_6_line_10.json : 10-Line Extended-Based Code Chunk 

Dataset_6_line_15.json : 15-Line Extended-Based Code Chunk 

Dataset_6_line_20.json : 20-Line Extended-Based Code Chunk 


## Implemtation Details
To run FunvVul and three baseline models, run FuncVul+Baselines.py file. If you are interested in knowing anything in detail email at sajal.csedu01@gmail.com

