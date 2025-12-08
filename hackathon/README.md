
## Demo  
Please see the MP4 for a quick demo.

---

## Multi-Agent System Workflow  

### 1. Supervisor Agent  
- At the center is our **Supervisor Agent** – think of it as the orchestral conductor. When a new ticket comes in, it coordinates all the other agents so they work together as one system.

---

### 2. Machine Learning Agent  
- Learned from historical dispatches which combinations of job, technician, and distance tend to succeed.  
- For every new dispatch, it predicts two things:  
  - The probability we fix it on the first visit.  
  - The probability it’s successful overall.  

---

### 3. Constraints Optimization Agent  
- Applies real-world rules:  
  - **Hard constraints**: Matching the right skill and ensuring the tech is available.  
  - **Soft factors**: Drive time, current workload, and success probabilities from the ML Agent.  
- Uses a combined score to select the best technician for each dispatch.  

---

### 4. UI + SQL Agent  
- The interface teams interact with.  
- Writes to backend tables and automatically triggers the first two agents whenever someone creates or updates a dispatch.  

---

### 5. Explanation Agent  
- Generates a plain-English summary:  
  - Explains why a technician was chosen.  
  - Covers factors like distance, skill, workload, and predicted success.  
- Ensures the system isn’t just a black box.  

---

### 6. Strategic Insights Agent  
- Looks across all assignments and provides business leaders with strategic insights:  
  - Which cities are short on certain skills.  
  - Signals where hiring or retraining may be needed.  

---
