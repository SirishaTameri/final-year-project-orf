from flask import Blueprint, render_template, request, jsonify
from src.services import predict_fraud
import logging
import time
import os
import pandas as pd
import io
from datetime import datetime

logger = logging.getLogger(__name__)
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('home.html', title="Dashboard")

@main.route('/predict')
def predict_page():
    return render_template('predict.html', title="Single Prediction")

@main.route('/compare')
def compare_page():
    return render_template('compare.html', title="Model Comparison")

@main.route('/batch')
def batch_page():
    return render_template('batch.html', title="Batch Analysis")

@main.route('/solutions')
def solutions_page():
    return render_template('solutions.html', title="Solutions")

@main.route('/resources')
def resources_page():
    return render_template('resources.html', title="Resources")

@main.route('/login')
def login_page():
    return render_template('login.html', title="Login")

@main.route('/signup')
def signup_page():
    return render_template('signup.html', title="Sign Up")

@main.route('/logout')
def logout():
    """Simple logout redirect"""
    return render_template('login.html', title="Login", message="You have been logged out.")

@main.route('/company')
def company_page():
    return render_template('company.html', title="About Us")

@main.route('/profile')
def profile_page():
    return render_template('profile.html', title="User Dashboard")

@main.route('/api/update-profile', methods=['POST'])
def update_profile():
    """Update user profile (Mock)"""
    try:
        data = request.get_json()
        return jsonify({
            'status': 'success',
            'message': 'Profile updated successfully',
            'user': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/api/subscribe', methods=['POST'])
def subscribe():
    """Handle newsletter subscription"""
    try:
        data = request.get_json()
        email = data.get('email')
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        return jsonify({
            'status': 'success',
            'message': f'Thank you! {email} has been subscribed to our newsletter.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/fraud-awareness')
def fraud_awareness_page():
    return render_template('fraud_awareness.html', title="Fraud Awareness")

@main.route('/compliance')
def compliance_page():
    return render_template('compliance.html', title="Legal & Compliance")

@main.route('/articles')
def articles_page():
    return render_template('articles.html', title="Knowledge Base")

# --- API Endpoints ---

@main.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Check if models exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'bert': os.path.exists(os.path.join(models_dir, 'bert_model')),
            'roberta': os.path.exists(os.path.join(models_dir, 'roberta_model'))
        }
    })

@main.route('/api/predict', methods=['POST'])
def predict():
    """Predict if a job posting is fraudulent"""
    try:
        start_time = time.time()
        
        data = request.get_json()
        text = data.get('text', '').strip()
        model_name = data.get('model', 'bert').lower()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model_name not in ['bert', 'roberta']:
            return jsonify({'error': 'Invalid model. Choose bert or roberta'}), 400
        
        result, error = predict_fraud(text, model_name)
        
        if error:
            return jsonify({'error': error}), 500
        
        processing_time = time.time() - start_time
        
        # Generate recommendation
        confidence = result['confidence']
        prediction = "LEGITIMATE" if not result['is_fraud'] else "FRAUDULENT"
        
        if prediction == "LEGITIMATE":
            if confidence > 0.8:
                recommendation = "This job posting appears highly legitimate. You can proceed with confidence."
            else:
                recommendation = "This job posting appears legitimate, but verify additional details before applying."
        else:
            if confidence > 0.8:
                recommendation = "⚠️ HIGH RISK: This job posting shows strong indicators of fraud. Avoid this opportunity."
            else:
                recommendation = "⚠️ This job posting shows some suspicious indicators. Exercise caution and verify thoroughly."
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'model': model_name.upper(),
            'processing_time': processing_time,
            'recommendation': recommendation,
            'is_fraud': result['is_fraud']
        })
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/api/compare', methods=['POST'])
def compare():
    """Compare predictions from both models"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        bert_result, bert_error = predict_fraud(text, 'bert')
        roberta_result, roberta_error = predict_fraud(text, 'roberta')
        
        if bert_error:
             return jsonify({'error': f'BERT Error: {bert_error}'}), 500
        if roberta_error:
             return jsonify({'error': f'RoBERTa Error: {roberta_error}'}), 500
        
        bert_pred = "LEGITIMATE" if not bert_result['is_fraud'] else "FRAUDULENT"
        roberta_pred = "LEGITIMATE" if not roberta_result['is_fraud'] else "FRAUDULENT"
        
        # Generate analysis
        analysis = ""
        if bert_pred == roberta_pred:
            if bert_pred == "LEGITIMATE":
                analysis = "Both models agree this job posting appears legitimate. High confidence in the assessment."
            else:
                analysis = "Both models agree this job posting shows strong indicators of fraud. Exercise extreme caution."
        else:
            if bert_result['confidence'] > roberta_result['confidence']:
                analysis = "Models disagree. BERT is more confident in its assessment. Consider additional verification."
            else:
                analysis = "Models disagree. RoBERTa is more conservative and flags potential fraud. Take precautions."
        
        return jsonify({
            'bert': {
                'prediction': bert_pred,
                'confidence': bert_result['confidence']
            },
            'roberta': {
                'prediction': roberta_pred,
                'confidence': roberta_result['confidence']
            },
            'analysis': analysis
        })
    
    except Exception as e:
        logger.error(f"Error in compare endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch predict multiple job postings from CSV file"""
    try:
        start_time = time.time()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        model_name = request.form.get('model', 'bert').lower()
        
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        if model_name not in ['bert', 'roberta']:
            return jsonify({'error': 'Invalid model. Choose bert or roberta'}), 400
        
        # Read the file content
        file_content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(file_content))
        
        # Look for text column
        text_column = None
        for col in df.columns:
            if col.lower() in ['text', 'description', 'job_text', 'posting', 'content']:
                text_column = col
                break
        
        if text_column is None:
            return jsonify({'error': 'CSV must contain a text column (text, description, job_text, posting, or content)'}), 400
        
        texts = df[text_column].fillna('').astype(str).tolist()
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 postings at a time'}), 400
        
        results = []
        legitimate_count = 0
        fraudulent_count = 0
        
        for i, text in enumerate(texts):
            if text.strip():  # Skip empty texts
                result, error = predict_fraud(text, model_name)
                if error:
                    results.append({
                        'text': text[:50] + '...' if len(text) > 50 else text,
                        'prediction': 'ERROR',
                        'confidence': 0.0,
                        'error': error
                    })
                else:
                    prediction = "LEGITIMATE" if not result['is_fraud'] else "FRAUDULENT"
                    results.append({
                        'text': text[:50] + '...' if len(text) > 50 else text,
                        'prediction': prediction,
                        'confidence': result['confidence']
                    })
                    if result['is_fraud']:
                        fraudulent_count += 1
                    else:
                        legitimate_count += 1
            else:
                results.append({
                    'text': '',
                    'prediction': 'EMPTY',
                    'confidence': 0.0
                })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'total_jobs': len(results),
            'legitimate_count': legitimate_count,
            'fraudulent_count': fraudulent_count,
            'processing_time': processing_time,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error in batch_predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500
