# Mini Proposal

MTUP: Multi-Task Unified Prompt Training for AMR Parsing

**Supervisor**: Trần Hưng Nghiệp, nghiepth@uit.edu.vn, 2026-04-03

## **1\. Mini Problem Formulation**

### **Project description**

We study whether AMR parsing can be improved by decomposing the generation process into multiple explicit supervised subtasks within a unified prompt. Instead of generating the final AMR graph directly, the model is trained to solve a sequence of easier subtasks, such as variable-free AMR generation, variable binding, concept extraction, or relation extraction, before producing the final graph. The main goal of the experiments is to provide controlled evidence for when and why prompt-based task decomposition improves semantic parsing.

### **Assumptions, constraints, scope, out of scope**

Assumptions:

* decoder-style LLMs can benefit from explicit intermediate supervision  
* AMR can be decomposed into meaningful sub-problems  
* the prompt structure can guide the model without changing architecture

Constraints:

* limited Vietnamese data  
* limited compute budget  
* limited project timeline

Scope:

* prompt-based supervised fine-tuning  
* Vietnamese first, then one cross-lingual evaluation  
* focus on parsing quality, parsability, and error type reduction

Out of scope:

* designing a new transformer architecture  
* large-scale synthetic corpus generation as the main contribution  
* downstream applications such as QA or retrieval

### **Existing approaches and research gap**

Direct AMR generation is simple but forces the decoder to solve concept prediction, relation prediction, graph well-formedness, and variable binding at the same time. The gap is a clean experimental study of prompt-based task decomposition for AMR parsing.

### **Research questions and hypotheses**

RQ1. Does MTUP outperform direct one-step AMR generation?  
H1. Yes, especially on parsability and structurally difficult cases.

RQ2. Which decomposition is most effective?  
H2. Decomposing into a small number of meaningful subtasks, such as no-var AMR and variable binding, will outperform both direct generation and overly long decompositions.

RQ3. Does the gain generalize beyond Vietnamese?  
H3. The gain will remain on English, though likely smaller than on Vietnamese.

RQ4. Why does MTUP help?  
H4. It reduces root errors, missing coordination roles, and variable/reference errors.

### **Objectives and potential contributions**

* Build a controlled benchmark for MTUP variants on Vietnamese  
* Identify which subtask decomposition is most useful  
* Provide cross-lingual evidence for generalizability  
* Provide detailed error analysis to explain improvements  
* Produce a paper-ready set of tables and ablations

## **2\. Related Work**

(Write later)

* **Direct seq2seq AMR parsing**  
   Simple and effective, but decoder must solve everything jointly.  
* **Variable-free or simplified AMR serialization**  
   Reduces difficulty, but often restores variables deterministically.  
* **Prompt-based or chat-style structured generation**  
   Flexible, but often lacks explicit supervision on intermediate representations.

## **3\. Data**

### **Vietnamese**

* `train_amr_1.txt`  
* `train_amr_2.txt`  
* `public_test_ground_truth.txt`

### **Cross-lingual dataset**

* **English AMR**: preferred  
* or one additional low-resource language

Do not start with synthetic data. Synthetic data should be a late ablation.

## **4\. Evaluation**

### **Real-life success**

A useful method should:

* produce valid AMR more consistently  
* reduce structural mistakes  
* generalize across languages

### **Experimental success**

Primary metric:

* **Smatch F1**

Secondary metrics:

* parsable output rate  
* exact bracket validity rate  
* root accuracy  
* concept F1  
* relation F1  
* reentrancy accuracy or variable-binding accuracy  
* selected targeted error counts:  
  * missing `:wiki`  
  * missing `:op1/:op2`  
  * missing `:manner`  
  * hallucinated concepts/tokens

### **Evaluation protocol**

* fixed train/dev/test split  
* report mean and std over 3 random seeds  
* greedy decoding for all main experiments  
* same post-processing across all methods  
* same evaluation script across all methods

## **5\. Baselines**

### **Baseline 1**

**Direct 1-task AMR generation**  
Input sentence → final AMR

### **Baseline 2**

**Direct variable-free AMR generation \+ deterministic variable restoration**  
Input sentence → variable-free AMR  
Then deterministic variable restoration

### **Baseline 3**

(Optional stronger baseline)  
**Chain-of-thought style verbal explanation prompt**  
Only if easy to run. Otherwise skip.

## **6\. Proposed Method**

### **Main idea**

Train the model with a Multi-Task Unified Prompt (MTUP) where multiple explicit supervised subtasks are placed sequentially in the same output. Each subtask has clear delimiters and a fixed format. The model first solves simpler semantic sub-problems, then generates the final graph or variable assignment.

### **Why it should work**

* reduces decoding difficulty  
* gives more explicit supervision  
* improves credit assignment during training  
* may reduce structural hallucination and reference errors

### **Main MTUP variants**

Keep only **three core variants**.

#### **MTUP-2**

Task 1: variable-free AMR  
Task 2: final AMR with variables (variable binding)

This is the main hypothesis test.

#### **MTUP-3A**

Task 1: concept list  
Task 2: variable-free AMR  
Task 3: final AMR with variables (variable binding)

This tests whether finer decomposition helps more.

#### **MTUP-3B**

Task 1: concept list  
Task 2: relation list  
Task 3: final AMR with variables (variable binding)

This tests another variant with explicit concept/relation decomposition.

#### **MTUP-4**

Task 1: concept list  
Task 2: relation list  
Task 3: variable-free AMR  
Task 4: final AMR with variables (variable binding)

This tests whether even finer decomposition helps more.

## **7\. Expected Results**

### **Expected findings**

* MTUP-2 beats direct 1-task baseline on Vietnamese  
* gains are strongest on parsability and structural error reduction  
* MTUP-3 and 4 variants may improve more, but may become slower  
* one of the decompositions should show whether “more tasks” truly helps  
* on English, gains may be smaller but still positive

### **Required experiments**

* one clean main table on Vietnamese  
* one ablation table isolating decomposition choice  
* one error analysis table  
* one reduced cross-lingual table

### **Backup plan if results are negative**

If MTUP does not improve Smatch, try:

* error analysis: parsable output rate, error types  
* focus and try to improve MTUP-2 only  
* evaluate only variable-free intermediate outputs  
* frame the result as a negative but informative finding on prompt decomposition limits
