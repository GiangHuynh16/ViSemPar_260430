# Experimental Plan

MTUP: Multi-Task Unified Prompt Training for AMR Parsing

**Supervisor**: Trần Hưng Nghiệp, nghiepth@uit.edu.vn, 2026-04-03

# **A. Core claim to test**

**MTUP helps AMR parsing because it decomposes one hard decoding problem into easier supervised sub-problems.**

So the experiments must answer three questions clearly:

1. Does decomposition help at all?  
   Compare 1-task vs 2-task vs 3-task, under the same model, same data, same training budget.  
2. Which decomposition helps most?  
   Compare:  
   * direct AMR  
   * no-var AMR \+ variable binding  
   * concepts \+ structure \+ AMR  
   * optional richer variants  
3. Does the idea generalize beyond Vietnamese?  
   First, show it on Vietnamese datasets. Then run the same small set on English, or one additional low-resource language if feasible.

# **B. Experimental policy**

For every main comparison:

* same base model  
* same train/dev/test split  
* same number of epochs or same total optimization steps  
* same decoding method  
* same maximum output budget policy  
* same preprocessing  
* same random seeds

Only one factor changes in each ablation.

# **C. Experimental Plan**

## **1\. Dataset**

(Add the data table with statistics)

* ### **Vietnamese dataset**

* ### **Cross-lingual dataset**

  * English AMR if possible  
  * Or one other low-resource AMR dataset

### **Split rule**

* Keep the public Vietnamese test set untouched.  
* Create one fixed validation split from the Vietnamese training data.  
* Use the same split for **all** experiments.

### **Data pipeline**

For every example, prepare these targets:

1. **Final AMR with variables**  
2. **Variable-free AMR**  
3. **Concept list**  
4. **Relation list**

### **Format rule**

Freeze one format for each target and never change it later.

## **2\. Evaluation Protocol**

### **Main metric**

* Smatch F1

### **Secondary metrics**

* parsable output rate  
* exact bracket validity rate  
* root accuracy  
* concept F1  
* relation F1  
* reentrancy / variable-binding accuracy

### **Targeted error counts**

Count how many predictions have:

* missing `:wiki`  
* missing `:op1/:op2`  
* missing `:manner`  
* hallucinated concepts or tokens

### **Evaluation rule**

* report mean ± std over 3 seeds  
* use greedy decoding for all main experiments  
* use the same post-processing and same evaluation script for all methods  
* choose checkpoint by validation Smatch, not by test score

## **3\. Baselines**

## **Baseline 1: Direct 1-task AMR**

**Goal:** standard direct generation baseline.

Input:

* sentence

Output:

* final AMR with variables

## **Baseline 2: No-var AMR \+ deterministic binding**

**Goal:** test whether removing variable burden helps even without learned binding.

Input:

* sentence

Output:

* variable-free AMR

Post-processing:

* restore variables with a fixed deterministic script

## **Baseline 3: Optional verbal-analysis baseline**

**Goal:** check whether longer prompting alone helps.

Run this only if time is enough.

Input:

* sentence

Output:

* short analysis  
* final AMR

## **4\. Proposed MTUP Methods**

## **MTUP-2**

**Purpose:** main hypothesis test.

Steps:

1. variable-free AMR  
2. final AMR with variables

## **MTUP-3A**

**Purpose:** test whether concept supervision helps.

Steps:

1. concept list  
2. variable-free AMR  
3. final AMR with variables

## **MTUP-3B**

**Purpose:** test whether concept \+ relation supervision helps.

Steps:

1. concept list  
2. relation list  
3. final AMR with variables

## **MTUP-4**

**Purpose:** test whether finer decomposition helps more.

Steps:

1. concept list  
2. relation list  
3. variable-free AMR  
4. final AMR with variables

## **5\. Prompt Templates**

## **Global rules**

### **System prompt**

Use exactly this for all methods:

You are an AMR parsing assistant. Follow the task exactly and output only the requested format.

### **Chat wrapper**

All training examples must follow this structure:

\<|im\_start|\>system  
{SYSTEM\_PROMPT}  
\<|im\_end|\>  
\<|im\_start|\>user  
{USER\_PROMPT\_BODY}  
\<|im\_end|\>  
\<|im\_start|\>assistant  
{TARGET\_OUTPUT}  
\<|im\_end|\>

### **Loss masking rule**

* Compute loss only on the assistant part.  
* Do not compute loss on system or user text.

### **Formatting rule**

* Keep section headers exactly identical across all examples.  
* Do not add explanations, comments, or extra text.  
* The assistant output must start immediately with the required section header.

## **1\. Baseline 1: Direct 1-task AMR generation**

### **User prompt body**

\#\#\# TASK  
Generate the final AMR graph for the input sentence.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[AMR\]

### **Assistant target**

\[AMR\]  
{FINAL\_AMR\_WITH\_VARIABLES}

### **Full Qwen2.5 template**

\<|im\_start|\>system  
You are an AMR parsing assistant. Follow the task exactly and output only the requested format.  
\<|im\_end|\>  
\<|im\_start|\>user  
\#\#\# TASK  
Generate the final AMR graph for the input sentence.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[AMR\]  
\<|im\_end|\>  
\<|im\_start|\>assistant  
\[AMR\]  
{FINAL\_AMR\_WITH\_VARIABLES}  
\<|im\_end|\>

## **2\. Baseline 2: Variable-free AMR generation**

### **User prompt body**

\#\#\# TASK  
Generate the AMR graph without variables for the input sentence.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[AMR\_NO\_VAR\]

### **Assistant target**

\[AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}

### **Full Qwen2.5 template**

\<|im\_start|\>system  
You are an AMR parsing assistant. Follow the task exactly and output only the requested format.  
\<|im\_end|\>  
\<|im\_start|\>user  
\#\#\# TASK  
Generate the AMR graph without variables for the input sentence.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[AMR\_NO\_VAR\]  
\<|im\_end|\>  
\<|im\_start|\>assistant  
\[AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}  
\<|im\_end|\>

## **3\. Baseline 3: Optional verbal-analysis baseline**

Run only if time allows.

### **User prompt body**

\#\#\# TASK  
Analyze the sentence briefly, then generate the final AMR graph.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[ANALYSIS\]  
\[AMR\]

### **Assistant target**

\[ANALYSIS\]  
{SHORT\_ANALYSIS}

\[AMR\]  
{FINAL\_AMR\_WITH\_VARIABLES}

### **Full Qwen2.5 template**

\<|im\_start|\>system  
You are an AMR parsing assistant. Follow the task exactly and output only the requested format.  
\<|im\_end|\>  
\<|im\_start|\>user  
\#\#\# TASK  
Analyze the sentence briefly, then generate the final AMR graph.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[ANALYSIS\]  
\[AMR\]  
\<|im\_end|\>  
\<|im\_start|\>assistant  
\[ANALYSIS\]  
{SHORT\_ANALYSIS}

\[AMR\]  
{FINAL\_AMR\_WITH\_VARIABLES}  
\<|im\_end|\>

# **MTUP Variants**

## **4\. MTUP-2**

### **User prompt body**

\#\#\# TASK  
Solve the AMR parsing problem in two steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: AMR\_NO\_VAR\]  
\[STEP 2: AMR\_WITH\_VAR\]

### **Assistant target**

\[STEP 1: AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}

\[STEP 2: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}

### **Full Qwen2.5 template**

\<|im\_start|\>system  
You are an AMR parsing assistant. Follow the task exactly and output only the requested format.  
\<|im\_end|\>  
\<|im\_start|\>user  
\#\#\# TASK  
Solve the AMR parsing problem in two steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: AMR\_NO\_VAR\]  
\[STEP 2: AMR\_WITH\_VAR\]  
\<|im\_end|\>  
\<|im\_start|\>assistant  
\[STEP 1: AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}

\[STEP 2: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}  
\<|im\_end|\>

## **5\. MTUP-3A**

### **User prompt body**

\#\#\# TASK  
Solve the AMR parsing problem in three steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: CONCEPTS\]  
\[STEP 2: AMR\_NO\_VAR\]  
\[STEP 3: AMR\_WITH\_VAR\]

### **Assistant target**

\[STEP 1: CONCEPTS\]  
{CONCEPT\_LIST}

\[STEP 2: AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}

\[STEP 3: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}

### **Full Qwen2.5 template**

\<|im\_start|\>system  
You are an AMR parsing assistant. Follow the task exactly and output only the requested format.  
\<|im\_end|\>  
\<|im\_start|\>user  
\#\#\# TASK  
Solve the AMR parsing problem in three steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: CONCEPTS\]  
\[STEP 2: AMR\_NO\_VAR\]  
\[STEP 3: AMR\_WITH\_VAR\]  
\<|im\_end|\>  
\<|im\_start|\>assistant  
\[STEP 1: CONCEPTS\]  
{CONCEPT\_LIST}

\[STEP 2: AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}

\[STEP 3: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}  
\<|im\_end|\>

## **6\. MTUP-3B**

### **User prompt body**

\#\#\# TASK  
Solve the AMR parsing problem in three steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: CONCEPTS\]  
\[STEP 2: RELATIONS\]  
\[STEP 3: AMR\_WITH\_VAR\]

### **Assistant target**

\[STEP 1: CONCEPTS\]  
{CONCEPT\_LIST}

\[STEP 2: RELATIONS\]  
{RELATION\_LIST}

\[STEP 3: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}

### **Full Qwen2.5 template**

\<|im\_start|\>system  
You are an AMR parsing assistant. Follow the task exactly and output only the requested format.  
\<|im\_end|\>  
\<|im\_start|\>user  
\#\#\# TASK  
Solve the AMR parsing problem in three steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: CONCEPTS\]  
\[STEP 2: RELATIONS\]  
\[STEP 3: AMR\_WITH\_VAR\]  
\<|im\_end|\>  
\<|im\_start|\>assistant  
\[STEP 1: CONCEPTS\]  
{CONCEPT\_LIST}

\[STEP 2: RELATIONS\]  
{RELATION\_LIST}

\[STEP 3: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}  
\<|im\_end|\>

## **7\. MTUP-4**

### **User prompt body**

\#\#\# TASK  
Solve the AMR parsing problem in four steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: CONCEPTS\]  
\[STEP 2: RELATIONS\]  
\[STEP 3: AMR\_NO\_VAR\]  
\[STEP 4: AMR\_WITH\_VAR\]

### **Assistant target**

\[STEP 1: CONCEPTS\]  
{CONCEPT\_LIST}

\[STEP 2: RELATIONS\]  
{RELATION\_LIST}

\[STEP 3: AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}

\[STEP 4: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}

### **Full Qwen2.5 template**

\<|im\_start|\>system  
You are an AMR parsing assistant. Follow the task exactly and output only the requested format.  
\<|im\_end|\>  
\<|im\_start|\>user  
\#\#\# TASK  
Solve the AMR parsing problem in four steps.

\#\#\# INPUT  
Sentence: {SENTENCE}

\#\#\# OUTPUT  
\[STEP 1: CONCEPTS\]  
\[STEP 2: RELATIONS\]  
\[STEP 3: AMR\_NO\_VAR\]  
\[STEP 4: AMR\_WITH\_VAR\]  
\<|im\_end|\>  
\<|im\_start|\>assistant  
\[STEP 1: CONCEPTS\]  
{CONCEPT\_LIST}

\[STEP 2: RELATIONS\]  
{RELATION\_LIST}

\[STEP 3: AMR\_NO\_VAR\]  
{AMR\_NO\_VAR}

\[STEP 4: AMR\_WITH\_VAR\]  
{FINAL\_AMR\_WITH\_VARIABLES}  
\<|im\_end|\>

### **NOTE: Fixed output formats for intermediate targets**

Students should use one fixed format for all training examples.

#### Concept list format

Recommended:

concept\_1 | want-01  
concept\_2 | boy  
concept\_3 | eat-01  
concept\_4 | apple

#### Relation list format

Recommended:

want-01 | :ARG0 | boy  
want-01 | :ARG1 | eat-01  
eat-01 | :ARG0 | boy  
eat-01 | :ARG1 | apple

#### Variable-free AMR format

Use one fixed format consistently, for example:

(want-01  
   :ARG0 (boy)  
   :ARG1 (eat-01  
       :ARG0 (boy)  
       :ARG1 (apple)))

or your chosen canonical no-var linearization.  
But once selected, do not change it.

### **NOTE: Recommendation for implementation**

To reduce bugs, students should implement each example using a small formatter like this:

* `build_user_prompt(method, sentence)`  
* `build_target_output(method, targets)`  
* `wrap_qwen_chat(system_prompt, user_prompt, assistant_target)`

This avoids manual prompt inconsistencies.

### **NOTE: Minimal checklist**

For every method, verify:

* same system prompt  
* same task wording style  
* same section-header style  
* assistant begins with first required header  
* no extra commentary  
* loss masked only on assistant tokens

## **6\. Ablation Plan**

## **Ablation 1: Main decomposition comparison**

Compare:

* Baseline 1  
* Baseline 2  
* MTUP-2  
* MTUP-3A  
* MTUP-3B  
* MTUP-4

**Why:** this is the main scientific comparison.

## **Ablation 2: Learned binding vs deterministic binding**

Compare:

* Baseline 2  
* MTUP-2

**Why:** isolates the value of learned variable binding.

## **Ablation 3: More subtasks or not**

Compare:

* MTUP-2  
* MTUP-3A  
* MTUP-3B  
* MTUP-4

**Why:** test whether more decomposition helps or becomes too long.

## **Ablation 4: Cross-lingual generalization**

Run only:

* Baseline 1  
* Baseline 2  
* best MTUP variant on Vietnamese

**Why:** keep cross-lingual evaluation small and focused.

## **Ablation 5: Optional synthetic-data ablation**

Run only after main results are complete.

Compare:

* best baseline, no synthetic  
* best baseline, with synthetic  
* best MTUP, no synthetic  
* best MTUP, with synthetic

**Why:** test whether synthetic data gives extra gain.

## **7\. Experiment Specifications**

## **Experiment 1: Vietnamese baselines**

Run:

* Baseline 1  
* Baseline 2

Output:

* baseline scores  
* parsable rate  
* average output length  
* error counts

## **Experiment 2: Vietnamese MTUP main results**

Run:

* MTUP-2  
* MTUP-3A  
* MTUP-3B  
* MTUP-4

Compare against:

* Baseline 1  
* Baseline 2

Output:

* main Vietnamese result table

## **Experiment 3: Vietnamese error analysis**

Analyze:

* root errors  
* missing `:wiki`  
* missing `:op1/:op2`  
* missing `:manner`  
* hallucinations

Output:

* one error table  
* 5 to 10 example cases

## **Experiment 4: Cross-lingual evaluation**

Run:

* Baseline 1  
* Baseline 2  
* best Vietnamese MTUP variant

Output:

* one reduced cross-lingual table

## **Experiment 5: Optional synthetic-data ablation**

Run only if the main story is complete.

## **8\. Execution Order**

### Step 1: Freeze data, splits, target formats, prompt templates.

### Step 2: Run Baseline 1 and MTUP-2, tune and show preliminary results.

### Step 3: Run all baselines, complete experiment 1\.

### Step 4: Run all MTUP variants, complete experiment 2\.

### Step 5: Produce the main result table and choose the best MTUP variant.

### Step 6: Run experiment 3, produce analysis tables.

### Step 7: Run additional experiments 4, 5\.

# Appendix

## **1\. Training Rules**

Students must keep the following identical across all main experiments:

* base model  
* tokenizer  
* LoRA setup  
* optimizer  
* learning rate  
* scheduler  
* number of epochs or total training steps  
* batch size  
* decoding settings  
* post-processing  
* evaluation script

### **Seed rule**

Run every main experiment with 3 seeds.

### **Checkpoint rule**

Choose the best checkpoint by validation Smatch only.

### **Logging rule**

For each run, save:

* config  
* seed  
* train loss  
* validation metrics  
* test predictions  
* average output length  
* failed examples

## **2\. Required Tables**

### **Table 1\. Main Vietnamese results**

Columns:

* Method  
* \#Tasks  
* Smatch  
* Parsable rate  
* Root accuracy  
* Concept F1  
* Relation F1  
* Avg output length

### **Table 2\. Decomposition ablation**

Columns:

* Method  
* Smatch  
* Parsable rate  
* Binding / reentrancy accuracy  
* Avg output length  
* Relative runtime

### **Table 3\. Error analysis**

Columns:

* Method  
* Root errors  
* Missing `:wiki`  
* Missing `:op1/:op2`  
* Missing `:manner`  
* Hallucination cases

### **Table 4\. Cross-lingual results**

Columns:

* Method  
* Language  
* Smatch  
* Parsable rate  
* Root accuracy

### **Table 5\. Optional synthetic-data ablation**

Columns:

* Method  
* Synthetic data  
* Smatch  
* Parsable rate

