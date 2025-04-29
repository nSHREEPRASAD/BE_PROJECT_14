import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Configuration
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
LORA_ADAPTERS = "Khalid02/fine_tuned_law_llama3_8b_lora-adapters"

# Global variables
model = None
tokenizer = None

def load_components():
    global model, tokenizer
    if model is None or tokenizer is None:
        with st.spinner("Loading model and tokenizer..."):
            try:
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )

                config = PeftConfig.from_pretrained(LORA_ADAPTERS)
                model = PeftModel.from_pretrained(
                    base_model,
                    LORA_ADAPTERS,
                    device_map="auto",
                    is_trainable=False
                )

                model = model.merge_and_unload()
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                raise

    return model, tokenizer

def generate_response(message, history, system_message, max_tokens, temperature, top_p):
    global model, tokenizer
    try:
        # Prepare conversation
        messages = [{"role": "system", "content": system_message}]
        for user_input, bot_response in history:
            if user_input:
                messages.append({"role": "user", "content": user_input})
            if bot_response:
                messages.append({"role": "assistant", "content": bot_response})
        messages.append({"role": "user", "content": message})
        
        # Format prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.1,
            use_cache=True,
        )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("🧑‍⚖️ Fine-tuned Llama 3.1 Legal Assistant")

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    system_message = st.sidebar.text_area("System Message", value="You are a legal expert chatbot. Provide accurate and helpful legal information.", height=100)
    max_tokens = st.sidebar.slider("Max New Tokens", min_value=1, max_value=2048, value=512)
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7)
    top_p = st.sidebar.slider("Top-p", min_value=0.1, max_value=1.0, value=0.95)

    # Reload model button
    if st.sidebar.button("🔄 Reload Model"):
        global model, tokenizer
        model, tokenizer = None, None
        load_components()

    load_components()

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        if user_input.strip() != "":
            response = generate_response(
                user_input,
                st.session_state.history,
                system_message,
                max_tokens,
                temperature,
                top_p,
            )
            st.session_state.history.append((user_input, response))

    # Chat History
    if st.session_state.history:
        for user_msg, bot_msg in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(bot_msg)

if __name__ == "__main__":
    main()