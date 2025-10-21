# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#-------------------------------------------------------------------------------------------#
# Model Architecture (same as before)
class GRUEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(GRUEncoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.GRU(input_size=n_features, hidden_size=self.hidden_dim, batch_first=True)
        self.rnn2 = nn.GRU(input_size=self.hidden_dim, hidden_size=embedding_dim, batch_first=True)

    def forward(self, x):
        x = x.reshape((-1, self.seq_len, self.n_features))
        x, _ = self.rnn1(x)
        x, hidden_n = self.rnn2(x)
        return hidden_n.reshape((-1, self.embedding_dim))

class GRUDecoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(GRUDecoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.GRU(input_size=input_dim, hidden_size=input_dim, batch_first=True)
        self.rnn2 = nn.GRU(input_size=input_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.output_layer(x)

class GRUAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(GRUAutoencoder, self).__init__()
        self.encoder = GRUEncoder(seq_len, n_features, embedding_dim)
        self.decoder = GRUDecoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#-------------------------------------------------------------------------------------------#
@st.cache_resource
def load_model():
    model = GRUAutoencoder(seq_len=140, n_features=1, embedding_dim=128)
    try:
        state_dict = torch.load("gru_v2.pth", map_location='cpu')
        # Handle key naming differences
        state_dict = {k.replace('gru', 'rnn') if 'gru' in k else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def predict_sample(model, sample_tensor):
    with torch.no_grad():
        reconstruction = model(sample_tensor.unsqueeze(0))  # Add batch dimension
        loss = torch.mean((reconstruction - sample_tensor.unsqueeze(0)) ** 2).item()
    return reconstruction.squeeze(0), loss

#-------------------------------------------------------------------------------------------#
# Streamlit App
st.set_page_config(page_title="ECJ Anomaly Detection", layout="wide")
st.title("🚨 ECJ Anomaly Detection using Autoencoder")

# Sidebar - CSV Upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with 140 features + 1 label column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate shape - should have 141 columns (140 features + 1 label)
        if df.shape[1] != 141:
            st.error(f"Error: CSV must contain 140 feature columns + 1 label column (got {df.shape[1]})")
        else:
            st.success(f"✅ Successfully loaded {df.shape[0]} samples with {df.shape[1]-1} features + 1 label")
            
            # Separate features and labels
            features = df.iloc[:, :-1].values.astype(np.float32)  # First 140 columns
            labels = df.iloc[:, -1].values  # Last column is the actual label
            
            # Convert to tensor
            data_tensor = torch.tensor(features).unsqueeze(-1)  # Shape: (N, 140, 1)
            
            # Tabs for layout
            tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Sample Analysis", "🚨 Batch Analysis"])
            
            with tab1:
                st.subheader("Data Overview")
                st.write("First 5 samples (features + label):")
                st.dataframe(df.head())
                st.write(f"Data shape: {df.shape}")
                
                # Show label distribution
                fig_dist, ax_dist = plt.subplots()
                df.iloc[:, -1].value_counts().plot(kind='bar', ax=ax_dist)
                ax_dist.set_title("Label Distribution (1=Normal, 0=Anomaly)")
                ax_dist.set_xlabel("Label")
                ax_dist.set_ylabel("Count")
                st.pyplot(fig_dist)
            
            with tab2:
                st.subheader("Individual Sample Analysis")
                sample_idx = st.selectbox("Select sample to analyze", range(len(df)), format_func=lambda x: f"Sample {x}")
                
                # Display selected sample features
                fig1, ax1 = plt.subplots(figsize=(12, 4))
                ax1.plot(features[sample_idx])
                ax1.set_title(f"Sample {sample_idx} - Features (Actual Label: {'Normal' if labels[sample_idx] == 1 else 'Anomaly'})")
                st.pyplot(fig1)
                
                # Get prediction for selected sample
                model = load_model()
                if model:
                    sample_tensor = data_tensor[sample_idx]  # Shape: (140, 1)
                    reconstruction, loss = predict_sample(model, sample_tensor)
                    
                    # Display reconstruction
                    fig2, ax2 = plt.subplots(figsize=(12, 4))
                    ax2.plot(reconstruction.numpy(), label='Reconstruction', color='orange')
                    ax2.plot(features[sample_idx], label='Original', alpha=0.6)
                    ax2.set_title(f"Reconstruction vs Original (Loss: {loss:.4f})")
                    ax2.legend()
                    st.pyplot(fig2)
                    
                    # Calculate threshold (using all samples)
                    with torch.no_grad():
                        reconstructions = model(data_tensor)
                        all_losses = torch.mean((reconstructions - data_tensor) ** 2, dim=(1, 2)).numpy()
                    threshold = 0.2197
                    
                    # Display prediction comparison
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Reconstruction Error", f"{loss:.6f}")
                    with col2:
                        pred = "Anomaly" if loss > threshold else "Normal"
                        st.metric("Predicted Label", pred)
                    with col3:
                        actual = "Normal" if labels[sample_idx] == 1 else "Anomaly"
                        st.metric("Actual Label", actual, 
                                 delta="Match" if (pred == actual) else "Mismatch",
                                 delta_color="normal" if (pred == actual) else "inverse")
                    
                    st.write(f"Threshold (95th percentile): {threshold:.6f}")
            
            with tab3:
                st.subheader("Batch Analysis - All Samples")
                model = load_model()
                
                if model:
                    with torch.no_grad():
                        reconstructions = model(data_tensor)
                        losses = torch.mean((reconstructions - data_tensor) ** 2, dim=(1, 2)).numpy()
                    
                    threshold =  0.2197
                    predictions = ["Anomaly" if l > threshold else "Normal" for l in losses]
                    actual_labels = ["Normal" if l == 1 else "Anomaly" for l in labels]
                    
                    results = pd.DataFrame({
                        "Sample": range(len(df)),
                        "Reconstruction Error": losses,
                        "Predicted Label": predictions,
                        "Actual Label": actual_labels,
                        "Match": [p == a for p, a in zip(predictions, actual_labels)]
                    })
                    
                    st.write("### Results for All Samples")
                    st.dataframe(results)
                    
                    # Calculate accuracy
                    accuracy = np.mean([p == a for p, a in zip(predictions, actual_labels)])
                    st.metric("Prediction Accuracy", f"{accuracy*100:.2f}%")
                    
                    # Download button
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Full Results", csv, "anomaly_results.csv", "text/csv")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis")