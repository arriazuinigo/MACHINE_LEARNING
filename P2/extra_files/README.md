1. Data Loading & Exploration - Joana
2. Preprocessing & Feature Analysis - Joana
3. K-Nearest Neighbours (KNN) - Laia
4. Logistic Regression - Andrea
5. Support Vector Machine (SVM) - Iñigo
6. Neural Network - Gabriel
7. Model Comparison & Evaluation - Todos
8. Written Report - Todos







# Execution complete: Extract PDF text content

Ran terminal command: import pdfplumber; pdf = pdfplumber.open('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf'); [print(p.extract_text()) for p in pdf.pages]; pdf.close()

Ran terminal command: import PyPDF2; r=PyPDF2.PdfReader('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf'); [print(p.extract_text()) for p in r.pages]

Ran terminal command:  pip install pdfplumber PyPDF2

Ran terminal command:  pip install pdfplumber PyPDF2

Ran terminal command:  python3 -m pip install pdfplumber PyPDF2

Ran terminal command: import pdfplumber; pdf = pdfplumber.open('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf'); [print(p.extract_text()) for p in pdf.pages]; pdf.close()

Ran terminal command:  cat "/Users/arriazui/Library/Application Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/07748a11-fac7-48bd-9659-8b98c83686ad/call_MHxja3Y2ZXhuSmowR000TjNoODg__vscode-1778581272792/content.txt"

Ran terminal command:  cat "/Users/arriazui/Library/Application Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/07748a11-fac7-48bd-9659-8b98c83686ad/call_MHxmdlZNZDZJOXZSZW9CZHlNTWE__vscode-1778581272793/content.txt"

Ran terminal command:  python3 -c "import pdfplumber; pdf = pdfplumber.open('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf'); print('\n'.join([p.extract_text() for p in pdf.pages if p.extract_text()])); pdf.close()" | head -n 50

Ran terminal command: import pdfplumber
pdf = pdfplumber.open('/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P2/extra_files/MHEDAS_Assignment2_SupervisedLearning-2.pdf')
for i, p in enumerate(pdf.pages):
    t = p.extract_text()
    if t:
        print(f'=== PAGE {i+1} ===')
        print(t)
pdf.close()

Read [](file:///Users/arriazui/Library/Application%20Support/Code/User/workspaceStorage/edf38277067f8be88c5314d78991c688/GitHub.copilot-chat/chat-session-resources/07748a11-fac7-48bd-9659-8b98c83686ad/toolu_bdrk_01VTbPFoEDiPjh7YhDqH4v3N__vscode-1778581272796/content.txt)

Now I have the full SVM task requirements. Let me create the file: 

Created [](file:///Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/python_code/Task5_SVM.py)

Now let's verify it runs cleanly: 

Execution complete: Run Task5_SVM.py and check for errors

Ran terminal command:  cd /Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/python_code && ls X_train_scaled.csv X_test_scaled.csv y_train.csv y_test.csv

Ran terminal command:  python3 Task2_Preprocessing.py && python3 Task5_SVM.py

Task5_SVM.py runs cleanly. Here's what it covers (matching all grading criteria):

**5.3 Kernel comparison** — trains Linear, RBF, and Polynomial SVMs with default C=1 and reports accuracy, AUC, and training time in a summary table + bar chart.

**5.4 RBF grid search** — sweeps C ∈ {0.01, 0.1, 1, 10, 100} × gamma ∈ {0.001, 0.01, 0.1, 1, 10}, produces two heatmaps (accuracy grid + training-time grid) and line plots showing the individual effect of each parameter.

**5.5 Final model** — RBF with C=10, gamma=0.001 (best from grid), classification report + confusion matrix saved.

Results:
| Kernel | Accuracy | AUC |
|---|---|---|
| Linear | 0.9737 | 0.9964 |
| RBF (default) | 0.9825 | 0.9950 |
| Polynomial | 0.9123 | 0.9954 |
| **RBF (tuned C=10, γ=0.001)** | **0.9825** | **0.9957** |

4 images saved: kernel comparison, grid search heatmap, C/gamma effect plots, and confusion matrix.

Made changes.