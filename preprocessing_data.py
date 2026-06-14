import pandas as pd
import sklearn
import joblib

headers = [
    # --- Network Identity (Usually dropped before modeling) ---
    'src_ip',          # Source IP address
    'src_port',        # Source port number
    'dst_ip',          # Destination IP address
    'dst_port',        # Destination port number

    # --- Protocol & Session Characteristics ---
    'protocol',        # Protocol type (TCP, UDP, etc.)
    'conn_state',      # Connection state (CON, FIN, etc.)
    'duration',        # Total session duration (in seconds)

    # --- Data Volume & Packets ---
    'src_to_dst_bytes', # Number of bytes from source to destination
    'dst_to_src_bytes', # Number of bytes from destination to source
    'src_ttl',          # Source Time-to-Live (TTL)
    'dst_ttl',          # Destination Time-to-Live (TTL)
    'src_loss',         # Packet loss from source
    'dst_loss',         # Packet loss from destination

    # --- Speed & Load Statistics ---
    'service_type',     # Service type (dns, http, smtp, etc.)
    'src_load_bps',     # Bits per second from source (Source Load)
    'dst_load_bps',     # Bits per second from destination (Dest Load)
    'src_packet_count', # Total packet count from source
    'dst_packet_count', # Total packet count from destination

    # --- TCP Technical Details ---
    'src_window_adv',   # Source window advertisement
    'dst_window_adv',   # Destination window advertisement
    'src_tcp_seq',      # Source TCP sequence number
    'dst_tcp_seq',      # Destination TCP sequence number
    'src_mean_packet_size', # Mean packet size from source
    'dst_mean_packet_size', # Mean packet size from destination

    # --- Content & Time ---
    'http_trans_depth', # HTTP transaction depth
    'http_res_body_len',# HTTP response body length
    'src_jitter',       # Source packet arrival time variation (jitter)
    'dst_jitter',       # Destination packet arrival time variation (jitter)
    'start_time',       # Start time timestamp
    'last_time',        # End time timestamp
    'src_inter_packet_time', # Inter-packet time from source (mms)
    'dst_inter_packet_time', # Inter-packet time from destination (mms)

    # --- Latency Metrics (Highly Critical for Forensics) ---
    'tcp_rtt',          # Total Round Trip Time (synack + ackdat)
    'tcp_synack_time',  # Time between SYN and SYN-ACK (connection setup)
    'tcp_ackdat_time',  # Time between SYN-ACK and ACK (data confirmation)

    # --- Contextual & Security Features ---
    'is_same_ip_port',  # Whether source & destination IP and Port are the same (1/0)
    'ct_state_ttl',     # Connection count based on state and TTL
    'ct_http_method',   # Count of HTTP method usage (GET/POST)
    'is_ftp_login',     # Whether this is an FTP login session (1/0)
    'ct_ftp_command',   # Count of commands in FTP session
    'ct_src_srv_count', # Count of connections for the same service from source
    'ct_dst_srv_count', # Count of connections for the same service to destination
    'ct_dst_ltm_count', # Count of connections to the same destination (last 100)
    'ct_src_ltm_count', # Count of connections from the same source (last 100)
    'ct_src_dport_count', # Count of connections to the same destination port from source
    'ct_dst_sport_count', # Count of connections from the same source port to destination
    'ct_src_dst_count', # Count of connections between the same source and destination

    # --- Classification ---
    'attack_category',  # Attack category name (if any)
    'label'             # 0 = Normal, 1 = Attack
]


def preprocess_data(transformer_path, scaler_path, ohc_path):
    url = "https://github.com/FatiBuuloloo/Dynamic_Thresholding_-Network_Traffic-/releases/download/dataset/UNSW-NB15_2.csv"
    df = pd.read_csv(url, names=headers, low_memory=False)
    print(f"""
    Columns : {df.columns}
    Shape : {df.shape}
    """)
    
    transformer = joblib.load(transformer_path)
    scaler = joblib.load(scaler_path)
    oh_encoder = joblib.load(ohc_path)

    print(f"Data has total features: {len(df.columns)}")

    print("================================")
    print("Dropping unnecessary columns...")
    print("================================")

    cols_to_drop = ['src_ip', 'dst_ip','start_time', 'last_time','src_port', 'dst_port','src_tcp_seq','dst_tcp_seq']
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Columns Left: {df.columns.tolist()}\n total feature:{len(df.columns)}\n") ## delete this later

    print("================================")
    print("Preprocessing columns 'service_type', 'ct_ftp_command', 'ct_http_method', 'is_ftp_login'")
    print("================================")
    df['service_type'] = df['service_type'].replace('-', 'unknown')
    df['ct_ftp_command'] = pd.to_numeric(df['ct_ftp_command'], errors='coerce')
    df['ct_ftp_command'] = df['ct_ftp_command'].fillna(0).astype(int)
    df[['ct_http_method','is_ftp_login']] = df[['ct_http_method','is_ftp_login']].fillna(0)
    print(f"Columns Left: {df.columns.tolist()}\n total feature:{len(df.columns)}\n") ## delete this later

    # making binary features from "ct_state_ttl", "ct_http_method", "is_ftp_login", "ct_ftp_command" columns
    print("================================")
    print("Creating binary features from 'ct_state_ttl', 'ct_http_method', 'is_ftp_login', 'ct_ftp_command'")
    print("================================")
    to_biner_columns = ["ct_state_ttl", "ct_http_method", "is_ftp_login", "ct_ftp_command"]
    df['has_state_ttl'] = (df['ct_state_ttl'] > 0).astype(int)
    df['is_http_request'] = (df['ct_http_method'] > 0).astype(int)
    df['has_ftp_login_attempt'] = (df['is_ftp_login'] > 0).astype(int)
    df['has_ftp_command'] = (df['ct_ftp_command'] > 0).astype(int)
    df.drop(columns=to_biner_columns, inplace=True)
    print(f"Columns Left: {df.columns.tolist()}\n total feature:{len(df.columns)}\n") ## delete this later

    print("Binary features created successfully.")
    
    # transforming data
    print("Transforming data...")
    yj__features = ['src_to_dst_bytes','dst_to_src_bytes','src_loss','dst_loss','src_packet_count','dst_packet_count',
                    'src_mean_packet_size','dst_mean_packet_size','http_res_body_len','http_trans_depth','ct_src_srv_count',
                    'ct_dst_srv_count','ct_dst_ltm_count','ct_src_ltm_count','ct_src_dport_count','ct_dst_sport_count',
                    'ct_src_dst_count', 'dst_load_bps','tcp_rtt','duration','dst_inter_packet_time','src_inter_packet_time',
                    'src_jitter','src_load_bps','tcp_synack_time','dst_jitter']
    minmax_feature = ['src_ttl','dst_ttl','src_window_adv','dst_window_adv']
    ohc_feature = ['protocol','conn_state','service_type']

    df[yj__features] = transformer.transform(df[yj__features])
    df[minmax_feature] = scaler.transform(df[minmax_feature])

    encoded_array = oh_encoder.transform(df[ohc_feature])
    new_col_names = oh_encoder.get_feature_names_out(ohc_feature)
    df_encoded_cols = pd.DataFrame(encoded_array, columns=new_col_names, index=df.index)
    df = pd.concat([df, df_encoded_cols], axis=1)
    df.drop(columns=ohc_feature, inplace=True)
    print(df.shape[1])

    
    return df.to_csv("preprocessed_data.csv", index=False)

if __name__ == "__main__":
    PATH_TRANSFORMER = "power_transformer_yj.pkl"
    PATH_SCALER = "minmax_scaler.pkl"  
    PATH_OHC = "one_hot_encoder.pkl"
    print("[START] Starting data preprocessing process...")
    df_hasil = preprocess_data(PATH_TRANSFORMER, PATH_SCALER, PATH_OHC)
    print("[FINISHED] Data preprocessing completed successfully.")