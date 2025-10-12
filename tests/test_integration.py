import pytest
import sys
import os
from unittest.mock import patch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    flask_app.config.update({
        "TESTING": True,
    })
    return flask_app

@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()

# Test 2.3.2.2: Response Structure
def test_check_endpoint_response_structure(client):
    """Test that the JSON response contains the required keys."""
    with patch('app.get_checker') as mock_get_checker:
        # Configure mock spelling checker
        mock_spelling = mock_get_checker.return_value
        mock_spelling.correct_spelling.return_value = ("test sentence", [])
        mock_spelling.check_grammar.return_value = []
        
        # Set up the mock to return different values based on argument
        def side_effect(checker_type):
            return mock_spelling
        mock_get_checker.side_effect = side_effect
        
        # Make request to the endpoint
        response = client.post('/check', data={'text': 'test'})
        
        # Verify response
        assert response.status_code == 200
        json_data = response.get_json()
        assert 'original_text' in json_data
        assert 'final_corrected_text' in json_data
        assert 'corrections' in json_data
        assert isinstance(json_data['corrections'], list)

# Test 2.3.2.3: Spelling Errors Only
def test_check_scenario_spelling_only(client):
    """Test handling text with only spelling mistakes."""
    with patch('app.get_checker') as mock_get_checker:
        # Create mocks for spelling and grammar checkers
        mock_spelling_checker = mock_get_checker.return_value
        
        # Configure spelling checker mock
        original_text = "This is a sentance with a mispeling."
        corrected_text = "This is a sentence with a misspelling."
        spelling_corrections = [
            {'original': 'sentance', 'correction': 'sentence', 'type': 'spelling', 'start': 10},
            {'original': 'mispeling', 'correction': 'misspelling', 'type': 'spelling', 'start': 28}
        ]
        
        # Set up the mock to return different values based on argument
        def side_effect(checker_type):
            if checker_type == 'spelling':
                mock_spelling_checker.correct_spelling.return_value = (corrected_text, spelling_corrections)
            elif checker_type == 'grammar':
                mock_spelling_checker.check_grammar.return_value = []
            return mock_spelling_checker
        
        mock_get_checker.side_effect = side_effect
        
        # Make request
        response = client.post('/check', data={'text': original_text})
        
        # Verify response
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['final_corrected_text'] == corrected_text
        assert len(json_data['corrections']) == 2
        assert all(c['type'] == 'spelling' for c in json_data['corrections'])

# Test 2.3.2.4: Grammar Errors Only
def test_check_scenario_grammar_only(client):
    """Test handling text with only grammar mistakes."""
    with patch('app.get_checker') as mock_get_checker:
        # Create mocks for spelling and grammar checkers
        mock_checker = mock_get_checker.return_value
        
        # Configure mocks
        original_text = "He go to the store."
        grammar_corrections = [
            {'original': 'go', 'correction': 'goes', 'type': 'grammar', 'start': 3}
        ]
        
        # Set up the mock to return different values based on argument
        def side_effect(checker_type):
            if checker_type == 'spelling':
                mock_checker.correct_spelling.return_value = (original_text, [])
            elif checker_type == 'grammar':
                mock_checker.check_grammar.return_value = grammar_corrections
            return mock_checker
        
        mock_get_checker.side_effect = side_effect
        
        # Make request
        response = client.post('/check', data={'text': original_text})
        
        # Verify response
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['final_corrected_text'] == original_text
        assert len(json_data['corrections']) == 1
        assert json_data['corrections'][0]['type'] == 'grammar'

# Test 2.3.2.5: Mixed Errors
def test_check_scenario_mixed_errors(client):
    """Test handling text with both spelling and grammar mistakes."""
    with patch('app.get_checker') as mock_get_checker:
        # Create mock
        mock_checker = mock_get_checker.return_value
        
        # Configure mocks
        original_text = "Shee go to the storee."
        spelling_corrected_text = "She go to the store."
        spelling_corrections = [
            {'original': 'Shee', 'correction': 'She', 'type': 'spelling', 'start': 0},
            {'original': 'storee', 'correction': 'store', 'type': 'spelling', 'start': 18}
        ]
        grammar_corrections = [
            {'original': 'go', 'correction': 'goes', 'type': 'grammar', 'start': 4}
        ]
        
        # Set up the mock to return different values based on argument
        def side_effect(checker_type):
            if checker_type == 'spelling':
                mock_checker.correct_spelling.return_value = (spelling_corrected_text, spelling_corrections)
            elif checker_type == 'grammar':
                mock_checker.check_grammar.return_value = grammar_corrections
            return mock_checker
        
        mock_get_checker.side_effect = side_effect
        
        # Make request
        response = client.post('/check', data={'text': original_text})
        
        # Verify response
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['final_corrected_text'] == spelling_corrected_text
        assert len(json_data['corrections']) == 3
        correction_types = [c['type'] for c in json_data['corrections']]
        assert 'spelling' in correction_types
        assert 'grammar' in correction_types

# Test 2.3.2.6: Correct Text
def test_check_scenario_correct_text(client):
    """Test handling text that is already correct."""
    with patch('app.get_checker') as mock_get_checker:
        # Create mock
        mock_checker = mock_get_checker.return_value
        
        # Configure mocks
        original_text = "This is a perfectly correct sentence."
        
        # Set up the mock to return different values based on argument
        def side_effect(checker_type):
            if checker_type == 'spelling':
                mock_checker.correct_spelling.return_value = (original_text, [])
            elif checker_type == 'grammar':
                mock_checker.check_grammar.return_value = []
            return mock_checker
        
        mock_get_checker.side_effect = side_effect
        
        # Make request
        response = client.post('/check', data={'text': original_text})
        
        # Verify response
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['final_corrected_text'] == original_text
        assert len(json_data['corrections']) == 0

# Test 2.3.2.7: Edge Cases
def test_check_scenario_edge_cases(client):
    """Test handling edge cases like empty string and whitespace."""
    # Test empty string
    response_empty = client.post('/check', data={'text': ''})
    assert response_empty.status_code == 400
    assert 'error' in response_empty.get_json()
    
    # Test whitespace only
    with patch('app.get_checker') as mock_get_checker:
        # Create mock
        mock_checker = mock_get_checker.return_value
        
        # Configure mocks
        whitespace_text = "   "
        
        # Set up the mock to return different values based on argument
        def side_effect(checker_type):
            if checker_type == 'spelling':
                mock_checker.correct_spelling.return_value = (whitespace_text, [])
            elif checker_type == 'grammar':
                mock_checker.check_grammar.return_value = []
            return mock_checker
        
        mock_get_checker.side_effect = side_effect
        
        # Make request
        response = client.post('/check', data={'text': whitespace_text})
        
        # Verify response
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['final_corrected_text'] == whitespace_text
        assert len(json_data['corrections']) == 0 