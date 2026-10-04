# ITC C508 – Prelim Reviewer
**Deep Learning (NLP) · Dr. Rodolfo C. Raga Jr. · Lectures 4–7**

> Based on the lecture PDFs in this repo (Lectures 4, 5, 6, 7). Lectures 1–3 are not included here.

---

## Lecture 4 – Introduction to LLMOps and RAG Concepts

### AI Assistants
- **AI assistants** (chatbots) are computer programs designed to **simulate conversations with human users**.
- Powered by **Natural Language Processing (NLP) algorithms**.
- Use **GPT-3** and other **Large Language Models (LLMs)** to understand and interpret human language, answer questions, and perform tasks.
- Used in customer service, education, and entertainment.

### GPT
- **GPT = Generative Pretrained Transformer**, developed by **OpenAI**; corresponds to the **right part (decoder) of the Transformer architecture**.
- Tasks: text completion, translation, summarization.
- GPT is a **next-token predictor**: it generates the next token based on probabilities, starting from a prompt. Each generated token is stored in "memory" with the prompt (**autoregressive generation**).
- Given a question **plus specific context**, it can produce an accurate answer.
- Its probabilistic approach **does not guarantee accuracy** → this is called **hallucination**.
- **GPT models are not really intelligent.** Their output is better described as **pattern recognition and completion based on prior data and statistics**.

### LLMOps
- **LLMOps (Large Language Model Operations)** is the set of practices and tools for **developing, deploying, and maintaining LLMs in production environments**.
- LLMs like **GPT-3, BERT, RoBERTa** revolutionized NLP, but they need significant compute and continuous monitoring.
- The term became popular with the **release of ChatGPT in 2022**.

### MLOps vs LLMOps
| Feature | MLOps | LLMOps |
|---|---|---|
| Scope | General practices for the entire ML model lifecycle | **Specialized subset of MLOps**, focused only on LLMs |
| Primary goal | Streamline and automate development, deployment, and maintenance of ML models | Address the unique challenges of developing and deploying LLMs in production |
| Key challenges | Version control, reproducibility, **data drift**, scalability | **Model size, high computational cost, prompt management, content safety and ethical guidelines** |

### 6 Components of LLMOps
1. **Data Collection and Preparation**
2. **Model Development**
3. **Prompt Engineering, RAG and Model Fine-tuning**
4. **Model Deployment**
5. **Observability**
6. **Reinforcement Learning from Human Feedback (RLHF)**

#### 1. Data Collection and Preparation
- A critical step for training or fine-tuning an LLM.
- Training from scratch needs both a **vast quantity and high quality** of data.
- Flow: Data Sources → (Collection & Preparation: needs vast storage, compute, effort) → **Dataset** → (Training: needs vast compute) → **Foundational Model**

#### 2. Model Development
- **Challenge:** Most organizations lack the compute and storage to train LLMs from scratch.
- **Solution:** Use **pre-trained LLMs via APIs**, or use the models directly.
- **How to choose an LLM:**
  - **Performance:** use leaderboards such as **Hugging Face's Open LLM Leaderboard** and **Stanford's HELM (Holistic Evaluation of Language Models)**.
  - **Business constraints:** **cost, token limits, model availability**.

#### 3. Prompt Engineering, RAG, Fine-tuning
- **Prompt Engineering:** the **iterative process of refining and optimizing inputs (prompts)** to guide an LLM toward a high-quality output.
  - **Zero-shot:** no examples in the prompt.
  - **Single-shot (one-shot):** one example of the desired output.
  - **Few-shot:** multiple examples.
  - Best practices: experiment with templates and phrasing; track prompt variations and settings like **temperature**; aim for the prompt that **consistently** gives the best output.
- **RAG (Retrieval-Augmented Generation):** grounds the LLM's response in **external, verifiable data** to improve relevance and reduce inaccuracies.
  - How it works:
    1. The LLM gets a query and is asked to answer it using the provided context.
    2. **Vector stores** keep the context relevant. They can be databases (**Pinecone, ChromaDB**) or libraries (**FAISS**).
    3. Vector stores keep data as **embeddings** and retrieve the most relevant information for the query.
    4. The original prompt and the retrieved context go to the LLM, which produces the final output.
- **Model Fine-tuning:** train a pre-trained LLM on a **smaller, task-specific dataset**. Use it when **prompt engineering and RAG are insufficient**.
  - Considerations: **Cost** (computationally expensive); the **black-box problem** (outputs can still be unpredictable).
  - Best practice: **versioning** (track fine-tuned model versions for updates and rollbacks).
  - Next step if it still fails: **train a custom model from scratch**.

#### 4. LLM Deployment
- Goal: deploy the model (built from scratch, fine-tuned, or open-source) for production.
- Challenges: **Model size**, **Compute power**.
- Metrics: **Latency** (time to generate a response); **Completion length** (longer output means higher latency).

#### LLM Life Cycle (diagram)
Select a Foundational Model → Prompt Engineering → Evaluation → **Good?**
- Not good → **Enough data?** Yes → **Fine-tuning**; No → **RAG** → loop back to evaluation
- Good → **Deploy to Production** → monitor **Observability, Data Drift, Model Drift**

---

## Lecture 5 – Developing a RAG-Based Chatbot Research Project
*Flow: problem • objectives • RAG design • experiment • evaluation*

### 1. Start with a Real Problem
- **WEAK, technology-first:** "I want to build a RAG chatbot." It doesn't say who has the problem or why it matters.
- **BETTER, problem-first:** "Users struggle to find reliable answers in many documents." It names a real need, can be measured, and justifies RAG.
- Then ask: **"Can RAG make this information easier to find and use?"**
- **The technology should follow the problem, not the other way around.**

### 2. Selecting a UN SDG (5 steps)
1. **Choose an SDG** → 2. **Identify a related subdomain** → 3. **Identify a real issue** → 4. **Define the specific problem** → 5. **Identify the RAG contribution**
- Start with the problem you care about; use the SDG to show its broader relevance.

### 3. Match the Problem to an SDG
| Example project | SDG |
|---|---|
| University information assistant | **SDG 4 – Quality Education** |
| Disaster-preparedness assistant | **SDG 11 – Sustainable Cities and Communities** |
| Public-service information assistant | **SDG 16 – Peace, Justice and Strong Institutions** |
| Health-information assistant | **SDG 3 – Good Health and Well-Being** |
- Choose the SDG the problem **directly supports**. The SDG gives context and relevance; **it does not prove the chatbot is effective.**

### 4. What Is RAG?
User Question → Retrieve Relevant Evidence → LLM + Retrieved Context → **Grounded Answer**
- Analogy: **RAG is an open-book exam.** Find the relevant page first, then answer.
- RAG **reduces** unsupported answers, but retrieval and generation **can still fail**.

### 5. The RAG Pipeline
- **A. Knowledge Base Preparation (offline / when documents update):** 1. Documents → 2. Chunks → 3. Embeddings → 4. Vector Store
- **B. Question Answering (real time):** 5. User Question → 6. **Retrieve Top-K** → 7. LLM + Context → 8. Grounded Answer
- Key idea: **retrieve the evidence first, then generate the answer from it.**

### 6. Research Methodology (6 steps)
Define problem → Prepare knowledge base → Develop RAG system → Run experiment → Evaluate performance → Analyze & explain results
- **System Development:** collect documents, build the retrieval pipeline, connect the LLM, create the interface.
- **Research Evaluation:** define test questions, compare configurations, measure retrieval and answer quality, analyze results.
- **Building the chatbot is the development task. Testing whether it works is the research task.**

### 7. Dataset in a RAG Project
- **A. Knowledge Base:** policies, manuals, FAQs, reports, approved webpages.
- **B. Test Question Set:** realistic questions, questions answerable from the KB, a few **unanswerable** ones, and expected answers or supporting passages.
- Record each document's **source, quantity, date, language, and permission status**.
- Prepare test questions **independently** from the chatbot being evaluated.

### 8. Research Objectives
- **General objective:** "To develop and evaluate a RAG-based chatbot that answers domain-specific questions using a curated knowledge base."
- **Specific objectives:** **Prepare** a domain-specific KB; **Develop** a RAG pipeline; **Evaluate** retrieval performance and answer quality; **Compare** two configurations or RAG vs. a non-RAG baseline.
- A strong objective links **PROBLEM → DEVELOPMENT → EVALUATION**.

### 9. Research Questions, Variables, and Evidence Must Align
| Research question | What you vary | What you measure |
|---|---|---|
| Does retrieval find the needed passage? | Top-K / retrieval setup | **Hit@K** / retrieval success |
| Are answers correct and relevant? | RAG vs. baseline / config | Correctness / relevance |
| Are answers supported by evidence? | Prompt or RAG setup | **Groundedness / faithfulness** |

### 10. A Simple, Fair Experiment
- Test **at least two design choices** (more choices earn more points):
  1. **Chunk size:** small (~300 tokens) vs. medium (~500 tokens)
  2. **Top-K:** K=3 vs. K=5
  3. **Embedding model:** compare two
  4. **RAG vs. non-RAG**
- **Keep constant:** same KB documents, same test questions, same LLM and prompt, same evaluation method and metrics, same temperature and key settings.
- **Change only ONE or TWO factors** so you can tell what made the difference.

### 11. Evaluation Criteria
| Aspect | Meaning | Measure |
|---|---|---|
| Retrieval performance | Was the supporting passage found? | **Hit@K or Recall@K** |
| Answer correctness | Is it factually right? | Human rating / reference comparison |
| Answer relevance | Does it address the question? | Human rating |
| Groundedness | Is it supported by retrieved evidence? | Evidence-supported rating |
| User evaluation | Useful and easy to use? | Survey (usefulness, ease, satisfaction) |
- **Don't rely on a single metric.**

### 12. Test Set
- Prepare **30–50 realistic questions** (a practical start for a senior undergraduate project).
- Include **easy, moderate, and difficult** questions.
- Cover **in-scope and out-of-scope** cases.
- Prepare **reference answers or passages**.
- Create the set **independently**. Don't let the chatbot generate its own evaluation questions.
- Examples: Easy: "University office hours?" Moderate: "Documents required for course transfer?" Difficult: "According to the 2023 policy, what are the grounds for academic probation?" Out-of-scope: "Who is the current mayor of Manila?"

### 13. One Clear Story
**Real Problem → SDG → Knowledge Base → RAG System → Testing & Evaluation → Findings & Contribution**
(Problem → Justified Solution → Development → Experiment → Results). This is what turns "we built a chatbot" into an **IT research project**.

---

## Lecture 6 – Building a Reliable Dataset for a RAG-Based Chatbot
*Documents → Chunks → Test Set*
**Required output:** a documented knowledge base plus an independent, evidence-backed test set.

### Two Components of a RAG Dataset
- **A. Knowledge Base:** the source documents the chatbot retrieves from (official policies, manuals, FAQs, research or technical documents, approved webpages).
- **B. Test-Question Set:** a separate set for evaluation. Each record has a **question, expected answer, supporting passage, and difficulty and scope labels**.
- **Core rule:** the knowledge base **supplies evidence**; the test set **measures whether the system uses that evidence correctly**.

### Why Dataset Reliability Matters (3 failure types)
1. **Source problem:** incorrect, outdated, duplicated, or unauthorized content gets into the KB.
2. **Retrieval problem:** the right information exists, but the retriever doesn't return the relevant chunk.
3. **Generation problem:** the LLM gets weak or ambiguous evidence and produces an unsupported answer.

### 6 Characteristics of a Reliable Knowledge Base
| | |
|---|---|
| **Relevant** | Supports the chatbot's stated purpose |
| **Authoritative** | Comes from a trustworthy or approved source |
| **Current** | Uses the correct version or effective date |
| **Complete** | Covers important topics without major gaps |
| **Traceable** | Source, processing, and version can be audited |
| **Retrievable** | Can be chunked and retrieved without losing meaning |
- Quality-control question: *"Can another researcher understand where each chunk came from and why it belongs?"*

### The Steps
1. **Define the chatbot's scope.**
   - In scope: enrollment requirements, deadlines, required documents, official procedures.
   - Out of scope: personalized legal advice, medical diagnosis, unverified rumors, unsupported information.
   - Student task: write 1 scope statement, 5 intended questions, and 3 questions the chatbot should redirect.
2. **Collect authorized documents:** **Identify** (the responsible office or authority) → **Verify** (official, current, appropriate) → **Record** (title, URL or path, date, version, access date) → **Preserve** (keep the original file).
   - **Publicly accessible does NOT automatically mean authorized for unrestricted reuse.**
3. **Create a document inventory:** Doc ID, Title, Version/Date, Language, Status, plus source organization, original URL or path, access date, authorization notes, processing status, optional checksum, relevance rationale.
   - The inventory is the **audit trail** of the knowledge base.
4. **Clean and inspect:** remove repeated headers and footers, page numbers inside sentences, OCR errors, duplicates, missing section boundaries, and badly extracted tables. Keep headings, check tables manually, record OCR fixes, and keep the original file.
   - **Rule: cleaning changes formatting and noise, not the meaning of the source.**
5. **Split into meaningful chunks:** each chunk should keep enough context to answer a question.
   - Chunk metadata: **Chunk ID, Document ID, section/heading, page number, character or token count, chunking strategy, overlap setting**.
   - Chunk size, overlap, and boundary strategy should be **tested, not assumed optimal**.
6. **Design an independent test set.** Question types:
   - **Direct factual:** the answer is in one passage.
   - **Multi-step:** combines facts or sections.
   - **Difficult:** comparison, multi-hop retrieval, interpretation.
   - **Unanswerable:** relevant topic, but the evidence is absent.
   - **Out of scope:** outside the chatbot's purpose.
   - **Minimum record:** Question ID, question, expected answer, evidence, difficulty, answerability, scope.
7. **Prepare reference answers and evidence** (e.g., "What documents are required?" → "Valid ID and completed form." → D002, p.4, §2 → "All required items included").
   - A **reference answer** summarizes what's expected. **Evidence** shows where that answer is supported.

### Knowledge-Base Preparation Pipeline
**Collect → Clean → Chunk → Embed → Index → Validate**
- **Freeze a version:** once evaluation starts, record the exact documents, chunks, embeddings, index, and configuration.

### Difficulty Levels
- **Easy:** one passage states the answer ("What documents are required?")
- **Moderate:** combine facts or interpret a section ("What is required before submitting the application?")
- **Difficult:** multi-hop, comparison, or reasoning ("Compare requirements for two applicant categories.")
- Assign difficulty with **explicit criteria**, not intuition.

### Scope and Answerability → Expected Behavior
| Type | Expected behavior |
|---|---|
| In scope + answerable | **Answer with evidence** |
| In scope + unanswerable | **State that the information is unavailable** |
| Out of scope | **Decline or redirect appropriately** |

### Data Leakage
- **Leakage** happens when **evaluation information influences the system before final testing**.
- Example: a student writes a test question → notices a failure and adds a special chunk containing the answer → the final score is **inflated**.
- **Research principle: the final test must stay independent of tuning decisions.**
- **How to prevent it (7 steps):**
  1. Define scope and collection rules
  2. Prepare and **freeze** the initial test set
  3. **Split development and evaluation questions**
  4. Tune using **development data only**
  5. **Lock** the final configuration
  6. **Evaluate once** on the untouched test set
  7. Record later changes as **new experiments**

### Leakage-Controlled Tuning
1. **Prepare the data:** e.g., 20 questions → **development set** (~15, D01–D15, used repeatedly for tuning) + **evaluation set** (~5, E01–E05, untouched, used **only once**). Freeze the evaluation set before tuning.
2. **Tune with development questions:** run → analyze errors → modify (cleaning, chunking, embedding model, retriever or reranker, prompt) → re-run, until satisfactory.
3. **Final evaluation:** lock the configuration → run E01–E05 once → calculate and report results.
- Changes made after the final evaluation count as **new experiments** (Experiment 2, 3…). Don't report them as the original evaluation.

### Fair Experiments
| Keep fixed | Change / compare | Measure / report |
|---|---|---|
| Test questions, rubric, evaluation procedure, task and scope, reporting format | Chunk size, chunk overlap, embedding model, retriever or reranker, prompt configuration | **Recall@k, Precision@k**, answer correctness, **faithfulness/groundedness**, failure examples |
- **Change one major factor at a time**, or justify a design that studies interactions.

### What to Document
- **Deliverable A – Knowledge Base:** inventory and provenance, original files and versions, cleaning log, chunking strategy and parameters, chunk metadata, embedding and vector-store settings, final KB version.
- **Deliverable B – Test Set:** independent questions, difficulty and answerability labels, reference answers, supporting passages and source IDs, rubric, dev/eval split, leakage-prevention record.
- **Assignment:** at least **20 manually authored test questions**.

### Key Takeaway
A reliable RAG dataset = **trustworthy sources** (relevant, authoritative, current, traceable) + **evaluation-ready questions** (answer, evidence, difficulty, scope) + **leakage-controlled experiments**. Reliable data makes experiments more defensible, but it **does not guarantee every answer is correct**.

---

## Lecture 7 – Data Chunking Strategies for RAG
*Documents → Chunks → Validate*

### RAG Begins with Retrieval, Not Generation
- Full pipeline: **Documents → Parsing → Chunking → Embeddings → Vector Database → Retrieval → LLM → Answer**
- Parsing through the Vector Database is the **indexing workflow**.
- **Chunking is the bridge between raw documents and searchable knowledge.**

### Why Prepare Documents Before Retrieval (4 reasons)
1. **Context limits:** LLMs and embedding models process bounded text; feeding a whole manual is inefficient or impossible.
2. **Retrieval precision:** a focused chunk matches the question better.
3. **Computational efficiency:** smaller units mean less embedding, storage, and processing work.
4. **Source fidelity:** keeps headings, pages, sections, tables, and metadata so answers can be traced.
- *"A 200-page PDF is a document. RAG needs retrievable evidence."*
- Think: **extract → clean → structure → chunk → index**

### Chunking Is an Indexing Decision (5 steps)
1. **Parse:** extract text, headings, tables, page info
2. **Clean:** remove repeated headers, footers, artifacts
3. **Chunk:** divide into units that can stand alone
4. **Embed:** convert each chunk into a vector
5. **Store:** keep vectors, chunk text, and metadata in a searchable KB
- **Poor chunks limit retrieval quality before the LLM even sees the question.**

### A Chunk Is a Retrieval Unit (4 goals)
1. **Self-contained meaning:** makes sense without the whole document
2. **Retrievable specificity:** contains distinctive information
3. **Context continuity:** relationships aren't cut at boundaries
4. **Traceability:** keeps metadata identifying its source
- **Chunk-size trade-off:** **too small** → context lost, ideas fragmented; **too large** → noise increases, retrieval blurs.
- **There is no universal best size.** It depends on document type, question type, model, and retrieval evaluation.

### Five Chunking Strategies
| # | Strategy | How | When / why |
|---|---|---|---|
| 01 | **Fixed-size** | Split by character, word, or token count | Simple baseline, easy to reproduce. Fast and uniform but **ignores natural breaks**; best for **simple logs** |
| 02 | **Overlapping** | Repeat a controlled portion between adjacent chunks (e.g., **10–20%** buffer) | Protects ideas that cross boundaries |
| 03 | **Sentence / Paragraph** | Split at natural language boundaries | Keeps readable units and local meaning |
| 04 | **Semantic** | Group adjacent content that stays conceptually related | Useful when **topics shift** within a document |
| 05 | **Structure-aware** | Follow headings, sections, chapters, tables, code | Strong fit for **manuals, policies, reports, technical documents** |

### Overlap Trade-off
- **Why use it:** a question may depend on information near a chunk boundary.
- **Trade-off:** more repeated content, embeddings, storage, and possible retrieval redundancy.
- **Starting point:** **~10–20%** overlap for general prose. It's a reasonable experiment, not a universal rule.

### Chunk by Meaning and Structure, Not Arbitrary Length
- Document → Section → Subsection → Content units (paragraphs, tables, code blocks)
- **Semantic/sentence** (natural language approach) vs. **structure-aware** (document structure approach)
- Result: chunks are **more coherent and easier to retrieve accurately**.

### Choose Strategy by Document Type
| Document type | Likely structure | Starting strategy | Why |
|---|---|---|---|
| Policy / manual | chapters → sections → rules | **Structure-aware** | Keeps rules with their context |
| Research paper | abstract → sections → paragraphs | **Paragraph + semantic** | Preserves argument flow |
| FAQ / knowledge base | question → answer blocks | **Paragraph / semantic** | Each answer can stand alone |
| Code / technical docs | classes → functions → blocks | **Structure-aware** | Avoids breaking executable units |
| Scanned PDF | page images → OCR text | **OCR + structure-aware** | Text must be recovered before chunking |
- **Principle: preserve the unit in which the source itself expresses meaning.**

### Chunking Parameters Are Experimental Settings
| Parameter | Default | Note |
|---|---|---|
| **Chunk size** | **~200–500 tokens** | Larger may suit dense technical passages |
| **Overlap** | **~10–20%** | Increase only when evaluation shows boundary loss |
| **Metadata** | **source + page + section** | Enables filtering, traceability, citations |
- **Context preservation:** keep headings, section labels, neighboring context, table relationships, and page references.
- **Don't optimize chunking by size alone. Optimize for retrieval usefulness.**

### Good Chunking Is a Retrieval-Quality Problem
- Loop: **Build** (baseline) → **Test** (representative questions) → **Inspect** (retrieved chunks) → **Refine** (size, overlap, structure)
- **Look for:** relevance, completeness, noise
- **Measure:** retrieval quality (precision, recall, hit rate), answer quality (correctness, groundedness, completeness), traceability
- **Key lesson:** chunking is **not a preprocessing detail you set once**. It's a **design variable to test** against the RAG task.

### Practical Chunking Workflow for PDF-based RAG (7 steps)
1. **Inspect the PDF:** text? tables? images? scanned pages?
2. **Extract & clean**
3. **Preserve structure:** headings, sections, pages, tables
4. **Choose strategy:** start simple, match the document
5. **Set parameters:** size, overlap, metadata
6. **Export chunks:** **JSON/JSONL** with source metadata
7. **Evaluate retrieval** with representative questions
- Mindset: start with a **transparent baseline**, and change it only when **retrieval evidence justifies it**.

### The Golden Rules of Chunking
1. **There is no universal strategy.** Match the method to document structure and query complexity.
2. **Balance size and semantics.** Large enough to keep meaning, small enough for precise retrieval.
3. **Iterate continuously.** Start with a baseline, measure keyword coverage and coherence, refine against real LLM responses.

---

## Quick Memory List
- GPT = **Generative Pretrained Transformer** (OpenAI, decoder side of the Transformer)
- Wrong but confident LLM output = **hallucination**
- LLMOps popularized by **ChatGPT (2022)**
- **6 LLMOps components:** Data → Model Dev → Prompt/RAG/Fine-tune → Deploy → Observability → RLHF
- LLM leaderboards: **Hugging Face Open LLM Leaderboard**, **Stanford HELM**
- Vector stores: **Pinecone, ChromaDB** (databases); **FAISS** (library)
- Prompting: **zero-shot / single-shot / few-shot**
- Deployment metrics: **latency, completion length**
- RAG analogy: **open-book exam**
- Test set size: **30–50** questions (L5); assignment: **≥20** manual questions (L6)
- Chunk size **200–500 tokens**; overlap **10–20%**
- KB pipeline: **Collect → Clean → Chunk → Embed → Index → Validate**
- Doc-collection steps: **Identify → Verify → Record → Preserve**
- Leakage = **evaluation info influences the system before final testing**
- Retrieval metrics: **Hit@K, Recall@K, Precision@K**; answer metrics: **correctness, relevance, groundedness/faithfulness**

---

## Practice Quiz

### Multiple Choice
1. GPT stands for:
   a) General Purpose Transformer b) Generative Pretrained Transformer c) Generated Predictive Text d) Graph Processing Transformer
2. GPT generating the next token from probabilities makes it a:
   a) classifier b) next-token predictor c) search engine d) rule-based system
3. When an LLM produces inaccurate information confidently, it is called:
   a) overfitting b) data drift c) hallucination d) leakage
4. LLMOps gained popularity with the release of ChatGPT in:
   a) 2020 b) 2021 c) 2022 d) 2023
5. Which is a key challenge of **LLMOps** but not typical of MLOps?
   a) Version control b) Prompt management c) Reproducibility d) Scalability
6. Which is a **library**, not a database, used as a vector store?
   a) Pinecone b) ChromaDB c) FAISS d) MySQL
7. A prompt with multiple examples is:
   a) zero-shot b) single-shot c) few-shot d) no-shot
8. Fine-tuning is used when:
   a) prompt engineering and RAG are insufficient b) there is no data c) latency is too low d) the model is too small
9. "RAG is like an ___ exam."
   a) closed-book b) oral c) open-book d) practical
10. Which SDG fits a university information assistant?
    a) SDG 3 b) SDG 4 c) SDG 11 d) SDG 16
11. In-scope but unanswerable question, expected behavior:
    a) Make up an answer b) State the information is unavailable c) Redirect to Google d) Answer from training data
12. Recommended starting overlap:
    a) 0% b) 10–20% c) 40–50% d) 75%
13. Best starting strategy for a policy/manual:
    a) Fixed-size b) Structure-aware c) Random d) Character-level
14. Best starting strategy for a scanned PDF:
    a) OCR + structure-aware b) Fixed-size c) Semantic only d) No chunking
15. A chunk that is too large causes:
    a) context lost b) noise increases and retrieval blurs c) ideas fragmented d) faster retrieval

### True or False
16. GPT models are truly intelligent and understand meaning like humans.
17. LLMOps is a specialized subset of MLOps.
18. Publicly accessible documents are automatically authorized for unrestricted reuse.
19. Cleaning a document may change its meaning to make it clearer.
20. There is no universal best chunk size.
21. The chatbot may generate its own evaluation questions.
22. The SDG proves that the chatbot is effective.
23. The evaluation set should be used only once, on the locked configuration.

### Identification
24. Practices and tools for developing, deploying, and maintaining LLMs in production.
25. When evaluation information influences the system before final testing.
26. The time it takes the model to generate a response.
27. Chunking strategy that splits by character, word, or token count.
28. Chunking strategy that groups adjacent content that stays conceptually related.
29. The document that serves as the audit trail of the knowledge base.
30. File format recommended for exporting chunks with metadata.

### Enumeration
31. The 6 components of LLMOps.
32. The 6 characteristics of a reliable knowledge base.
33. The 5 chunking strategies.
34. The KB preparation pipeline (6 steps).
35. The 4 steps for collecting authorized documents.
36. The 3 failure types (why dataset reliability matters).
37. The 4 goals of a chunk.
38. The 3 golden rules of chunking.

### Answer Key
1. b  2. b  3. c  4. c  5. b  6. c  7. c  8. a  9. c  10. b  11. b  12. b  13. b  14. a  15. b
16. False  17. True  18. False  19. False (cleaning changes formatting and noise, not meaning)  20. True  21. False  22. False  23. True
24. LLMOps  25. Data leakage  26. Latency  27. Fixed-size chunking  28. Semantic chunking  29. Document inventory  30. JSON/JSONL
31. Data Collection & Preparation; Model Development; Prompt Engineering, RAG & Fine-tuning; Model Deployment; Observability; RLHF
32. Relevant, Authoritative, Current, Complete, Traceable, Retrievable
33. Fixed-size, Overlapping, Sentence/Paragraph, Semantic, Structure-aware
34. Collect, Clean, Chunk, Embed, Index, Validate
35. Identify, Verify, Record, Preserve
36. Source problem, Retrieval problem, Generation problem
37. Self-contained meaning, Retrievable specificity, Context continuity, Traceability
38. No universal strategy; Balance size and semantics; Iterate continuously
