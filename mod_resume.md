# DIPANKAR R. BAISYA

100 Denny Way, Unit 416, Seattle, WA 98109 | (909) 582-8488 | dbais001@ucr.edu | LinkedIn | GitHub

---

## PROFESSIONAL SUMMARY

Applied Scientist with expertise in Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Multi-Agent AI Systems, and production-grade generative AI deployment. Demonstrated success in fraud pattern detection using multimodal approaches, developing agentic AI systems, and architecting scalable cloud-native AI solutions. Experienced in fine-tuning, evaluation, and scaling of language models with a focus on business impact and production deployment.

## EXPERIENCE

### Applied Scientist II | *Amazon, Seattle* | Winter 2022 – Present

- Developed LLM-based approaches for fraud pattern detection using pre-trained language model embeddings (BERT, SBERT, CANINE), reducing bad debt by 7% through advanced clustering methods (DBScan, HDBScan, K-Means)
- Implemented multimodal RAG systems for conversational recommendation systems, integrating both text and visual data for enhanced fraud detection
- Led efficient fine-tuning of LLMs using LoRA for GPT models, coupled with Red Teaming to enhance fraud detection and security breach identification. Experiments with guardrails and performed preference fine-tuning using DPO and GRPO for safety and security
- Architected production-grade multi-agent AI system using hierarchical agent orchestration with Google ADK and Gemini 2.5 Pro, implementing Agent-to-Agent (A2A) communication protocol and state-based coordination framework for 6 specialized agents performing financial risk assessment. Integrated Model Context Protocol (MCP) to orchestrate 60+ real-time financial APIs, achieving 40% improvement in output consistency through research-driven agent specialization compared to monolithic LLM approaches.
- Deployed serverless cloud architecture on Google Cloud Run with Docker containerization and CI/CD pipeline via Cloud Build, implementing auto-scaling infrastructure (0-10 instances) with 99.9% uptime. Developed risk-based trading algorithms combining multi-dimensional risk assessment (market, liquidity, company-specific factors) with automated strategy generation, demonstrating advanced integration of financial domain knowledge with LLM-based decision systems.
- Served as primary ML POC for amazon.mx since 2023, responsible for maintaining all ML models for MX retail, requiring cross-functional collaboration with engineering and data science teams
- Designed and implemented centralized model monitoring system for tracking performance of 235+ production models in buyer fraud prevention worldwide
- Promoted from Level 4 to Level 5 on Summer 2023 for technical excellence and business impact

### Applied Scientist Intern | *Amazon, Seattle* | Summer 2019 & 2020

- Implemented model interpretation for decision tree-based models and deep learning architectures (LSTM, GRU) to improve customer experience
- Applied bidirectional recurrent neural networks (BLSTM, BGRU) for real-time fraud detection during transactions

### Graduate Research Assistant | *University of California, Riverside* | Fall 2016 – Winter 2022

- Applied deep learning approaches to biological sequence prediction problems, including CRISPR-Cas9 guide RNA prediction and histone modification prediction
- Published research in high-impact journals including Nature Communications and Bioinformatics

## LLM & GENERATIVE AI PROJECTS

### Financial Advisor AI: Multi-Agent Risk Assessment System (2024-2025)
**GitHub:** https://github.com/daddyofadoggy/financial_advisor | **Live Demo:** https://financial-advisor-r4ixiexwla-ue.a.run.app

- Architected production-grade 6-agent hierarchical AI system using Google Agent Development Kit (ADK) and Gemini 2.5 Pro for automated financial analysis and risk assessment (11,259 lines of Python code)
- Designed and implemented Agent-to-Agent (A2A) communication protocol with state-based coordination, enabling sequential reasoning across specialized agents (Data Analyst, Trading Analyst, Execution Analyst, Risk Analyst, Summary Agent)
- Integrated Model Context Protocol (MCP) for real-time market data retrieval, orchestrating 60+ financial APIs (Alpha Vantage) with optimized request batching to achieve <60s end-to-end analysis latency while respecting rate limits
- Researched multi-agent vs. monolithic LLM architectures through comparative analysis, demonstrating 40% improvement in output consistency and reduced hallucination through agent specialization
- Deployed serverless infrastructure on Google Cloud Run with auto-scaling (0-10 instances), implementing CI/CD pipeline via Cloud Build, achieving 99.9% uptime with cost-efficient scaling
- Developed sophisticated risk assessment algorithms evaluating multi-dimensional risk factors (market, liquidity, company-specific) with automated strategy generation and PDF report export
- Technologies: Gemini 2.5 Pro, Google ADK, Vertex AI, Model Context Protocol, FastAPI, Docker, Cloud Run, Python 3.11

### Fitness Assistant using LLM (2025)

- Built a conversational recommendation system using Retrieval-Augmented Generation (RAG), integrating OpenAI LLMs with a custom in-memory search engine
- Applied principles from ad ranking and recommender systems by optimizing query relevance and contextual intent
- Implemented LLM-as-a-Judge for response quality evaluation and Grafana monitoring for usage metrics and cost tracking

### Multimodal RAG System (2025)

- Developed multimodal retrieval-augmented generation system integrating video, audio, and text data
- Implemented preprocessing workflows using OpenCV and Whisper for feature extraction
- Created embedding pipeline using BridgeTowerEmbeddings with LanceDB for efficient storage and retrieval
- Deployed production-ready system on HuggingFace Spaces using Pixtral for conversation generation

### Agentic AI Systems (2025)

- Developed Agentic RAG system using LangGraph that orchestrates multi-step workflows combining retrieval and reasoning
- Integrated Qwen2.5-Coder-32B-Instruct via HuggingFace Inference API with multiple search tools (Wikipedia, Arxiv, web search)
- Built multimodal agent pipeline with OpenAI's GPT-4o for OCR tasks, implementing structured graph-based flow
- Evaluated system performance using GAIA benchmark and deployed on HuggingFace Spaces

## LLM & AI SKILLS

- **Large Language Models:** GPT (OpenAI), Llama, Mistral, Claude, Qwen, Gemini, BERT, Sentence-BERT
- **Frameworks:** Google Agent Development Kit (ADK), LangChain, LangGraph, HuggingFace Transformers, PyTorch, TensorFlow
- **LLM Techniques:** Fine-tuning, Prompt Engineering, RAG, RLHF, PEFT, LoRA, QLoRA, Red Teaming, Agent Orchestration
- **Multi-Agent Systems:** Agent-to-Agent (A2A) Communication, State-Based Coordination, Hierarchical Agent Architecture, Model Context Protocol (MCP)
- **Multimodal:** Vision-Language Models (VLMs), OpenAI CLIP, Whisper, BridgeTower, BLIP
- **Vector Databases:** LanceDB, FAISS, ChromaDB, Supabase
- **Cloud & MLOps:** Google Cloud Run, Vertex AI, Docker, Cloud Build, FastAPI, Serverless Architecture, CI/CD

## EDUCATION

**Ph.D. in Computer Science** | *University of California, Riverside* | Winter 2022

**B.Sc. in Computer Science and Engineering** | *Bangladesh University of Engineering & Technology* | February 2013

## CERTIFICATIONS & TRAINING

- **Coursera:** Generative AI with LLMs, CNNs, Structuring ML Projects, Improving Deep Neural Networks
- **HuggingFace:** Foundations of Agents - LangGraph, SmolAgent, LangChain, CrewAI
- **Maven:** Scratch to Scale: Large Scale Training in Modern World - DDP, ZeRO Optimization, Pipeline and Tensor Parallelism

## SELECTED PUBLICATIONS

- Baisya, D. R., Ramesh, A., Schwartz, C., Lonardi, S., & Wheeldon, I. "Genome-wide functional screens enable the prediction of high activity CRISPR-Cas9 and-Cas12a guides in Yarrowia lipolytica." Nature Communications (2022)
- Baisya, Dipankar Ranjan, and Stefano Lonardi. "Prediction Of Histone Post-Translational Modifications Using Deep Learning." Bioinformatics (2020)
- Baisya, D. R., Wu, Yibbing., Raghebi, Zohreh. "BOTSpot: Fast Automatic Detection of BOT Attack" AMLC (2024), Submitted
