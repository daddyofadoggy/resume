# DIPANKAR R. BAISYA

100 Denny Way, Unit 416, Seattle, WA 98109 | (909) 582-8488 | dbais001@ucr.edu | [LinkedIn](https://www.linkedin.com/in/dbaisya) | [GitHub](https://github.com/daddyofadoggy)

---

## PROFESSIONAL SUMMARY

Applied Scientist with expertise in Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Multi-Agent AI Systems, and production-grade generative AI deployment. Demonstrated success in fraud pattern detection using multimodal approaches, developing agentic AI systems, and architecting scalable cloud-native AI solutions. Experienced in fine-tuning, evaluation, and scaling of language models with a focus on business impact and production deployment.

## EXPERIENCE

### Applied Scientist II | *Amazon, Seattle* | Winter 2022 – Present


- Served as primary ML POC for amazon.mx since 2023, responsible for maintaining all ML models of MX retail, requiring cross-functional collaboration with engineering and data science teams
- Developed LLM-based approaches for fraud pattern detection using pre-trained language model embeddings (BERT, SBERT, CANINE), reducing bad debt by 7% through advanced clustering methods (DBScan, HDBScan, K-Means)
- Implemented multimodal RAG systems for conversational recommendation systems, integrating both text and visual data for enhanced fraud detection
- Led efficient fine-tuning of LLMs using LoRA for GPT models, coupled with Red Teaming to enhance fraud detection and security breach identification. Experimented with guardrails and performed preference fine-tuning using DPO and GRPO to implement safety and security measures
- Developed multi-agent risk-based trading algorithms combining multi-dimensional risk assessment (market, liquidity, company-specific factors) with Model Context Protocol (MCP) to orchestrate 60+ real-time financial APIs. Achieved 20% improvement in output consistency through research-driven agent specialization compared to monolithic LLM approaches.
- Experimented with different techniques to optimize model inference time and scale up prod models e.g., Quantization, knowledge Distilation, Fused Kernal and Optimizer, Distributed training, Mixed Precision training (FP16/BF16) etc
- Promoted from Level 4 to Level 5 on Summer 2023 for technical excellence and business impact

### Applied Scientist Intern | *Amazon, Seattle* | Summer 2019 & 2020

- Implemented model interpretation for decision tree-based models and deep learning architectures (LSTM, GRU) to improve customer experience
- Applied Auto-encoder and bidirectional recurrent neural networks (BLSTM, BGRU) for real-time fraud detection during transactions

### Graduate Research Assistant | *University of California, Riverside* | Fall 2016 – Winter 2022

- Applied deep learning approaches to biological sequence prediction problems, including CRISPR-Cas9 guide RNA prediction and histone modification prediction
- Published research in high-impact journals including Nature Communications and Bioinformatics

## AI PROJECTS

### RiskNavigator AI: Multi-Agent Financial Advisor (2025)
**GitHub:** https://github.com/daddyofadoggy/financial_advisor | **Live Demo:** https://financial-advisor-r4ixiexwla-ue.a.run.app

- Architected production-grade 6-agent hierarchical AI system implementing Agent-to-Agent (A2A) communication protocol and Model Context Protocol (MCP) to orchestrate 60+ real-time financial APIs for comprehensive automated financial analysis and risk assessment
- Researched multi-agent vs. monolithic LLM architectures through comparative analysis, demonstrating 20% improvement in output consistency and reduced hallucination through agent specialization
- Deployed serverless cloud architecture with auto-scaling infrastructure (0-10 instances) achieving 99.9% uptime, delivering multi-dimensional risk assessment (market, liquidity, company-specific) with automated strategy generation and PDF report export


### Multimodal RAG System (2025)
**Live Demo:** https://huggingface.co/spaces/doggdad/multimodal-rag

- Developed multimodal retrieval-augmented generation system integrating video, audio, and text data for cross-modal understanding
- Implemented preprocessing workflows using OpenCV for video processing and Whisper for audio transcription and feature extraction
- Created embedding pipeline using BridgeTowerEmbeddings with LanceDB vector database for efficient multimodal storage and retrieval
- Deployed production-ready system on HuggingFace Spaces using Pixtral vision-language model for multimodal conversation generation

### Agentic RAG System (2025)

- Developed Agentic RAG system using LangGraph that orchestrates multi-step workflows combining retrieval and reasoning capabilities
- Integrated Qwen2.5-Coder-32B-Instruct via HuggingFace Inference API with multiple specialized tools including Wikipedia search, Arxiv search, web search via Tavily, and mathematical operations
- Implemented Supabase vector database for semantic similarity search and intelligent question retrieval
- Evaluated system performance using GAIA benchmark and deployed production system on HuggingFace Spaces

### Build and Deploy GPT-2 from Scratch (2025)
**GitHub:** https://github.com/daddyofadoggy/InstructGPT-from-scratch | **Live Demo:** https://huggingface.co/spaces/doggdad/InstructGPTFinetuned | **Blog:** https://daddyofadoggy.github.io/blog/posts/LLM-From-Scratch/

- Implemented complete GPT architecture from scratch (124M parameters) including byte-pair encoding tokenization, multi-head attention, layer normalization, GELU activations, and residual connections, then executed full pretraining pipeline and validated against OpenAI's pretrained weights up to 1,558M parameters
- Fine-tuned model for spam classification achieving 97% accuracy, developed instruction-following capabilities using Alpaca-style prompts with automated evaluation, and implemented Direct Preference Optimization (DPO) for LLM alignment to human preferences
- Developed controlled text generation system with temperature scaling and top-k sampling, deployed production chatbot on HuggingFace Spaces with comprehensive technical blog post documenting end-to-end LLM development lifecycle

## LLM & AI SKILLS

- **Large Language Models:** GPT (OpenAI), Llama, Mistral, Claude, Qwen, Gemini, BERT, Sentence-BERT, Pixtral
- **Frameworks:**  HuggingFace Transformers, LangChain, LangGraph,  PyTorch, TensorFlow, Google Agent Development Kit (ADK),
- **Distributed Training:** FSDP, DeepSpeed, Megatron, Torchtitan, Mixed Precision Training (FP16/BF16)
- **LLM Techniques:** Fine-tuning, Prompt Engineering, RAG, RLHF, PEFT, LoRA, QLoRA, Red Teaming, Agent Orchestration, DPO, GRPO
- **Multi-Agent Systems:** Agent-to-Agent (A2A) Communication, State-Based Coordination, Hierarchical Agent Architecture, Model Context Protocol (MCP)
- **Multimodal:** Vision-Language Models (VLMs), OpenAI CLIP, Whisper, BridgeTower, BLIP
- **Vector Databases:** LanceDB, FAISS, ChromaDB, Supabase
- **Cloud & MLOps:** Amazon AWS, Lambda AI, Modal, Google Cloud Run, Vertex AI, Docker, Cloud Build, FastAPI, Serverless Architecture,  

## EDUCATION

**Ph.D. in Computer Science** | *University of California, Riverside* | Winter 2022

**B.Sc. in Computer Science and Engineering** | *Bangladesh University of Engineering & Technology* | February 2013

## CERTIFICATIONS & TRAINING

- **Coursera:** Generative AI with LLMs, CNNs, Structuring ML Projects, Improving Deep Neural Networks
- **HuggingFace:** Foundations of Agents - LangGraph, SmolAgent, LangChain, CrewAI
- **Maven:** Scratch to Scale: Large Scale Training in Modern World - DDP, ZeRO Optimization, Pipeline and Tensor Parallelism, MoE, Mixed Precision training (FP16/BF16)

## SELECTED PUBLICATIONS

- Baisya, D. R., Ramesh, A., Schwartz, C., Lonardi, S., & Wheeldon, I. "Genome-wide functional screens enable the prediction of high activity CRISPR-Cas9 and-Cas12a guides in Yarrowia lipolytica." Nature Communications (2022)
- Baisya, Dipankar Ranjan, and Stefano Lonardi. "Prediction Of Histone Post-Translational Modifications Using Deep Learning." Bioinformatics (2020)
- Baisya, D. R., Wu, Yibbing., Raghebi, Zohreh. "BOTSpot: Fast Automatic Detection of BOT Attack" AMLC (2024), Submitted
