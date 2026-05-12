# P1: Assigment 1: Unsupervised Learning Methods

## Required Python Libraries

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- joblib
- threadpoolctl
- mrmr-selection

### Install all required libraries with:

```shell
python3 -m pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib threadpoolctl mrmr-selection
```


### certificate problem for a dataset 
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context



for creating the reporting: 
https://prism.openai.com/?u=667b8e68-9a44-4942-b97f-42e516794f5d&pg=1&m=main.tex



# REQUIREMENTS TASK 5: 
Now I have both the PDF requirements for Task 5 and the Task5_SVM.py file. Let me analyze whether the file accomplishes all Task 5 requirements.

Task 5 — Support Vector Machine (SVM) requirements:

Train SVM classifiers with at least three kernels: linear, RBF, and polynomial.

Report accuracy and training time for each kernel in a summary table.

For the RBF kernel, explore the effect of C and gamma. Plot a small grid of results.

Fit a final SVM with your chosen kernel and settings. Print the classification report and confusion matrix.

The grading rubric awards 7 points for kernel comparison with accuracy and training time, 5 points for the final model with justification, and 3 points for discussing the hyperparameter effects of C and gamma.

Looking at what Task5_SVM.py actually implements, it trains SVM models with linear, RBF, and polynomial kernels and reports their accuracy and training times.

It also performs a grid search across C values from 0.01 to 100 and gamma values from 0.001 to 10 for the RBF kernel, then visualizes the results with heatmaps showing how accuracy and training time vary across these parameter combinations.

The script fits a final SVM model using the best hyperparameters found, generates a classification report and confusion matrix for evaluation, and plots the confusion matrix for visualization.

The implementation looks solid—it even includes AUC reporting beyond what's required, and all the key deliverables like the confusion matrix and final summary table are present.

I'm noticing a potential path issue though: the output directory is hardcoded to a P1 location, but the CSV files are actually in the P2 python_code directory, so I need to verify the paths align correctly. I'm noticing the hardcoded output directory path points to a non-existent top-level python_code folder, when it should be targeting the P2/python_code directory where the script actually lives. The workspace only contains P1 and P2 subdirectories, so this path configuration is definitely incorrect and would cause the outputs to save to the wrong location. I'm noticing a path mismatch—the Task5_SVM.py file is configured to write output to /Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/python_code, but the actual CSV data files and the script itself are located in the P2/python_code/ subdirectory, so the output directory path needs to be adjusted to match where the data actually lives. I've verified all the requirements are met, including the three kernels, grid search analysis, heatmaps, and the final model with classification report and confusion matrix. The main issue I'm seeing is that output_dir is hardcoded to a specific user path, which will break when someone else runs the script—it should use os.path.dirname(os.path.abspath(__file__)) instead to dynamically find the correct directory. So the real question is whether Task5_SVM.py actually meets all the requirements from the PDF, regardless of where the files are located. Looking at it functionally, yes—it does handle all the required tasks, though there's definitely that path issue with the hardcoded output directory that needs fixing.


Read [](file:///Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/python_code/Task5_SVM.py#1-1), lines 1 to 100



Read [](file:///Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/python_code/Task5_SVM.py#100-100), lines 100 to 250

Read [](file:///Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/python_code/Task5_SVM.py#250-250), lines 250 to 399

Now let me read the PDF to check the Task 5 requirements: 

Execution complete: Extract PDF content

Ran terminal command: import pdfplumber; pdf = pdfplumber.open('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf'); [print(page.extract_text()) for page in pdf.pages]

Ran terminal command:  cat "/Users/arriazui/Library/Application Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/ba23c930-13e8-4bc2-9d99-dc8f6e214c22/call_MHxRQ2ZaWEVtOERaVmE1TXlEQ2o__vscode-1778581272864/content.txt"

Ran terminal command:  cat "/Users/arriazui/Library/Application Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/ba23c930-13e8-4bc2-9d99-dc8f6e214c22/call_MHxiSVJDbmNZeXFVVlNGbUNRWkQ__vscode-1778581272865/content.txt"

Ran terminal command:  cat "/Users/arriazui/Library/Application Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/ba23c930-13e8-4bc2-9d99-dc8f6e214c22/call_MHxOODVQZTYySjlySWFNcUFsRlE__vscode-1778581272866/content.txt"

Ran terminal command:  head -n 200 "/Users/arriazui/Library/Application Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/ba23c930-13e8-4bc2-9d99-dc8f6e214c22/call_MHxQSEFsZWlHeUJQTHRpSmJvT2g__vscode-1778581272867/content.txt"

Ran terminal command:  cat "/Users/arriazui/Library/Application Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/ba23c930-13e8-4bc2-9d99-dc8f6e214c22/call_MHx1Vml4YnYxZnRyRGtQQ0w3dmU__vscode-1778581272868/content.txt"

Ran terminal command:  python3 -c "import pdfplumber; pdf = pdfplumber.open('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf'); print('\n'.join([page.extract_text() for page in pdf.pages]))" | head -n 100

Ran terminal command: import pdfplumber
pdf = pdfplumber.open('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf')
for i, page in enumerate(pdf.pages):
    txt = page.extract_text()
    if txt:
        print(f'=== PAGE {i+1} ===')
        print(txt)

Read [](file:///Users/arriazui/Library/Application%20Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/ba23c930-13e8-4bc2-9d99-dc8f6e214c22/toolu_bdrk_01L9VZRWNwpVDTV8JFRZCmYx__vscode-1778581272871/content.txt)

Now I have everything I need. Here's the analysis:

---

## Task 5 Requirements vs. Task5_SVM.py

### ✅ All task requirements are met:

| Requirement | Status | Details |
|---|---|---|
| Three kernels (linear, RBF, polynomial) | ✅ | `kernel_configs` in §5.3 |
| Accuracy & training time per kernel (summary table) | ✅ | `kernel_df` printed + bar chart saved |
| C × gamma grid search for RBF | ✅ | 5×5 grid in §5.4 |
| Plot grid results | ✅ | Dual heatmaps (accuracy + time) + line plots for C and gamma effects |
| Final model with classification report | ✅ | §5.5 — `classification_report()` printed |
| Confusion matrix for final model | ✅ | `ConfusionMatrixDisplay` printed & saved |
| Kernel/hyperparameter justification | ✅ | Inline comments in §5.5 explain the choice |

### ⚠️ One bug to fix — hardcoded `output_dir`:

Line 35 sets:
```python
output_dir = '/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/python_code'
```

This path does **not** exist in the workspace. The CSV data files (`X_train_scaled.csv`, etc.) live in python_code. The correct path should use `__file__`:

```python
output_dir = os.path.dirname(os.path.abspath(__file__))
```

Want me to fix that path?