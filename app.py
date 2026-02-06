"""
Streamlit UI for RAG System for System Logs.
Enhanced interface with Tabs, Chatbot History, and robust Log Ingestion.
"""
import streamlit as st
from src.pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Log Q&A System",
    page_icon="🤖",
    layout="wide"
)

# Initialize pipeline in session state
if 'pipeline' not in st.session_state:
    with st.spinner("Initializing RAG Pipeline (Checking GPU/API)..."):
        st.session_state.pipeline = RAGPipeline()
    st.success("Pipeline initialized!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🤖 Log Analyst Assistant")
st.markdown("""
**Intelligent Log Analysis System (RAG)**  
Supported: Linux (Syslog), Windows (Event Logs), macOS.  
*Multi-Question Planning, Token Tracking.*
""")

# Sidebar for statistics & Configuration
with st.sidebar:
    st.header("📊 System Status")
    stats = st.session_state.pipeline.get_stats()
    
    # Enhanced Device Status
    device_color = "green" if "cuda" in stats["device"] else "red"
    device_icon = "🚀" if "cuda" in stats["device"] else "🐢"
    st.markdown(f"**Compute:** :{device_color}[{stats['device'].upper()} {device_icon}]")
    
    # Collection Status
    db_info = st.session_state.pipeline.vector_db.get_collection_info()
    st.metric("Indexed Chunks", db_info["count"], help="Total log segments in current active collection")
    
    st.divider()
    
    # Collection Manager
    st.subheader("📚 Knowledge Base")
    
    # List collections (with persistence check logic)
    collections = st.session_state.pipeline.list_collections()
    if not collections:
        collections = ["log_chunks"]
    
    # Current selection
    current_coll_name = st.session_state.pipeline.vector_db.collection_name
    
    # Ensure current is in list (handle newly created but not yet physically in DB)
    if current_coll_name and current_coll_name not in collections:
        collections.append(current_coll_name)
    
    # Selector
    selected_coll = st.selectbox(
        "Active Collection", 
        collections, 
        index=collections.index(current_coll_name) if current_coll_name in collections else 0,
        key="coll_selector"
    )
    
    # Handle switch
    if selected_coll != current_coll_name:
        st.session_state.pipeline.switch_collection(selected_coll)
        st.session_state.messages = [] # Clear chat on switch
        st.rerun()

    # Create new
    with st.expander("New Collection"):
        new_coll_name = st.text_input("Name", placeholder="e.g. server_A_logs", key="new_coll_input")
        if st.button("Create"):
            if new_coll_name and new_coll_name.strip():
                clean_name = new_coll_name.strip().replace(" ", "_").lower()
                st.session_state.pipeline.switch_collection(clean_name)
                st.session_state.messages = [] # Clear chat on switch
                st.success(f"Switched to {clean_name}")
                st.rerun()
            else:
                st.error("Invalid name")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("⚠️ Wipe Database", type="secondary"):
        st.session_state.pipeline.clear_database()
        st.session_state.messages = []
        st.success("Database cleared!")
        st.rerun()

# Main Tabs
tab1, tab2 = st.tabs(["💬 Chat & Analysis", "📤 Log Ingestion"])

# --- TAB 1: CHAT INTERFACE ---
with tab1:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If assistant, show extra details if stored
            if message["role"] == "assistant" and "details" in message:
                details = message["details"]
                # Sources
                if details.get("sources"):
                    with st.expander(f"📚 Evidence ({len(details['sources'])} chunks)"):
                        for i, chunk in enumerate(details['sources'], 1):
                            st.markdown(f"**Chunk {i}** (Score: {chunk.get('score', 0):.2f})")
                            st.code(chunk['text'], language='text')
                            st.caption(f"File: {chunk.get('source_file', 'unknown')} | {chunk.get('start_time', '')} - {chunk.get('end_time', '')}")
                            st.divider()
                
                # Stats
                if details.get("usage"):
                    usage = details["usage"]
                    with st.expander("📊 Token Usage"):
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Total In", usage.get("total_input_tokens", 0))
                        s2.metric("Total Out", usage.get("total_output_tokens", 0))
                        s3.metric("Max In", usage.get("max_input_tokens", 0))
                        s4.metric("Max Out", usage.get("max_output_tokens", 0))
                        st.caption(f"Calls: {usage.get('total_calls', 0)}")

    # Chat Input
    if prompt := st.chat_input("Ask a question about the logs... (or multiple separated by ?)"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Multi-question logic
            raw_questions = [q.strip() + "?" for q in prompt.split("?") if q.strip()]
            if not raw_questions and prompt.strip():
                raw_questions = [prompt.strip()]

            # Container for accumulated response
            full_response_container = st.container()
            
            with st.spinner("Analyzing logs..."):
                for i, sub_q in enumerate(raw_questions, 1):
                    # Visual separator if multiple questions
                    if len(raw_questions) > 1:
                        st.markdown(f"#### ❓ Q{i}: {sub_q}")

                    result = st.session_state.pipeline.run(sub_q)
                    answer = result["answer"]
                    sources = result["sources"]
                    usage = result.get("usage_stats", {})
                    
                    st.markdown(answer)
                    
                    # Store details for history
                    msg_details = {"sources": sources, "usage": usage}
                    
                    # Append strictly to history (one message per sub-question to keep it granular? 
                    # OR one big message? Let's do one big message for the session, but display incrementally)
                    # Actually, for history state, let's append ONE assistant message containing all answers 
                    # But that makes "details" hard to structure.
                    # BETTER APPROACH: Append separate assistant messages for each Q if there are multiple.
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (f"**Q{i}:** {answer}" if len(raw_questions) > 1 else answer),
                        "details": msg_details
                    })
                    
                    # Show details immediately for this run
                    if sources:
                        with st.expander(f"📚 Evidence (Q{i})"):
                            for k, chunk in enumerate(sources, 1):
                                st.code(chunk['text'], language='text')
                    
                    if usage and usage.get("total_calls", 0) > 0:
                         with st.expander(f"📊 Stats (Q{i})", expanded=False):
                            s1, s2, s3, s4 = st.columns(4)
                            s1.metric("Total In", usage['total_input_tokens'])
                            s2.metric("Total Out", usage['total_output_tokens'])
                            s3.metric("Max In/Call", usage.get('max_input_tokens', 0))
                            s4.metric("Max Out/Call", usage.get('max_output_tokens', 0))
                            st.caption(f"Number of LLM Calls: {usage['total_calls']}")
                    
                    if i < len(raw_questions):
                        st.divider()

# --- TAB 2: INGESTION ---
with tab2:
    st.header("Upload System Logs")
    
    c1, c2 = st.columns(2)
    with c1:
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "log", "csv"])
    with c2:
        pasted_text = st.text_area("Or paste text", height=150)

    # Log Type Selector
    log_type = st.radio(
        "Log Type",
        ["System Logs (Linux/Windows/macOS)", "Container Logs (Docker/Kubernetes)"],
        index=0,
        horizontal=True
    )
    
    # Map friendly name to internal code
    log_type_code = "system" if "System" in log_type else "container"

    if st.button("🚀 Ingest Logs", type="primary"):
        target_text = None
        source_name = "manual_paste"
        
        if uploaded_file:
            target_text = uploaded_file.read().decode("utf-8")
            source_name = uploaded_file.name
        elif pasted_text.strip():
            target_text = pasted_text
        
        if target_text:
            progress_text = "Operation in progress. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            with st.status("Processing Pipeline...", expanded=True) as status:
                st.write(f"1. Preprocessing logs ({log_type_code} mode)...")
                # Simulate steps or just call valid pipeline
                st.write(f"2. Chunking text (Time-aware)...")
                st.write("3. Generating embeddings (GPU Acceleration)...")
                
                result = st.session_state.pipeline.ingest_logs(target_text, source_name, log_type=log_type_code)
                
                if result["success"]:
                    status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                    st.success(result["message"])
                    st.balloons()
                    # Trigger reload to update stats in sidebar
                    # st.rerun() # Verify if this disrupts the tab view - often better to just show success
                else:
                    status.update(label="❌ Ingestion Failed", state="error")
                    st.error(result["message"])
            
            my_bar.empty()
        else:
            st.warning("Please provide log data.")
