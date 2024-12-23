import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

file_name = "TestbedSunJun13Flows"

def parse_xml_to_dataframe(xml_file):
    tree = ET.parse(rf"labeled_flows_xml\{xml_file}")
    root = tree.getroot()
    data = []
    for testbed in root.findall(f'.//{file_name}'):
        record = {
            'appName': testbed.find('appName').text if testbed.find('appName') is not None else None,
            'totalSourceBytes': int(testbed.find('totalSourceBytes').text) if testbed.find('totalSourceBytes') is not None else 0,
            'totalDestinationBytes': int(testbed.find('totalDestinationBytes').text) if testbed.find('totalDestinationBytes') is not None else 0,
            'totalDestinationPackets': int(testbed.find('totalDestinationPackets').text) if testbed.find('totalDestinationPackets') is not None else 0,
            'totalSourcePackets': int(testbed.find('totalSourcePackets').text) if testbed.find('totalSourcePackets') is not None else 0,
            'direction': testbed.find('direction').text if testbed.find('direction') is not None else '',
            'sourceTCPFlagsDescription': testbed.find('sourceTCPFlagsDescription').text if testbed.find('sourceTCPFlagsDescription') is not None else '',
            'destinationTCPFlagsDescription': testbed.find('destinationTCPFlagsDescription').text if testbed.find('destinationTCPFlagsDescription') is not None else '',
            'source': testbed.find('source').text if testbed.find('source') is not None else '',
            'protocolName': testbed.find('protocolName').text if testbed.find('protocolName') is not None else '',
            'sourcePort': int(testbed.find('sourcePort').text) if testbed.find('sourcePort') is not None else 0,
            'destination': testbed.find('destination').text if testbed.find('destination') is not None else '',
            'destinationPort': int(testbed.find('destinationPort').text) if testbed.find('destinationPort') is not None else 0,
            'startDateTime': testbed.find('startDateTime').text if testbed.find('startDateTime') is not None else '',
            'stopDateTime': testbed.find('stopDateTime').text if testbed.find('stopDateTime') is not None else '',
            'Tag': testbed.find('Tag').text if testbed.find('Tag') is not None else '',
            'sourcePayloadAsBase64': testbed.find('sourcePayloadAsBase64').text if testbed.find('sourcePayloadAsBase64') is not None else '',
            'destinationPayloadAsBase64': testbed.find('destinationPayloadAsBase64').text if testbed.find('destinationPayloadAsBase64') is not None else '',
            'destinationPayloadAsUTF': testbed.find('destinationPayloadAsUTF').text if testbed.find('destinationPayloadAsUTF') is not None else '',
            'sourcePayloadAsUTF': testbed.find('sourcePayloadAsUTF').text if testbed.find('sourcePayloadAsUTF') is not None else '',
        }
        data.append(record)
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    categorical_columns = [
        'appName', 'direction', 'sourceTCPFlagsDescription',
        'destinationTCPFlagsDescription', 'source', 'protocolName',
        'destination', 'sourcePayloadAsBase64', 'destinationPayloadAsBase64',
        'destinationPayloadAsUTF', 'sourcePayloadAsUTF'
    ]
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    encoder = OrdinalEncoder()
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    df['startDateTime'] = pd.to_datetime(df['startDateTime'], errors='coerce')
    df['stopDateTime'] = pd.to_datetime(df['stopDateTime'], errors='coerce')
    df['startTimestamp'] = df['startDateTime'].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    df['stopTimestamp'] = df['stopDateTime'].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    df = df.drop(columns=['startDateTime', 'stopDateTime'])
    df.dropna(inplace=True)
    numeric_columns = [
        'totalSourceBytes', 'totalDestinationBytes', 'totalDestinationPackets',
        'totalSourcePackets', 'sourcePort', 'destinationPort',
        'startTimestamp', 'stopTimestamp'
    ]
    df[numeric_columns] = df[numeric_columns].astype(float)
    print(df.info())
    return df

def train_and_evaluate(df):
    X = df.drop(columns=['Tag'])
    y = df['Tag']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print("Cross-validation - F1 score per fold:", cv_scores)
    print("Cross-validation - Average F1 score:", np.mean(cv_scores))
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return model

if __name__ == "__main__":
    xml_file = f"{file_name}.xml"
    df = parse_xml_to_dataframe(xml_file)
    df = preprocess_data(df)
    trained_model = train_and_evaluate(df)
