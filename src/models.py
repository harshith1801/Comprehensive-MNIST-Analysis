import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates an SVM classifier.
    Returns a dictionary with performance metrics.
    """
    print("Training SVM...")
    start_time = time.time()
    
    clf = SVC(kernel='rbf', gamma='scale', C=1, random_state=42)
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"SVM training finished in {train_time:.2f} seconds.")
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': train_time,
        'confusion_matrix': cm
    }

def train_and_evaluate_knn(X_train, y_train, X_test, y_test, k=5):
    """
    Trains and evaluates a KNN classifier.
    Returns a dictionary with performance metrics.
    """
    print(f"Training KNN with k={k}...")
    start_time = time.time()
    
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"KNN training finished in {train_time:.2f} seconds.")
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': train_time
    }
