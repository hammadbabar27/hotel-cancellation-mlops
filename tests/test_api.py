from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
EXAMPLE = {
 'no_of_adults': 2, 'no_of_children': 0,
 'no_of_weekend_nights': 1, 'no_of_week_nights': 2,
 'type_of_meal_plan': 'Meal Plan 1', 'required_car_parking_space': 0,
 'room_type_reserved': 'Room_Type 1', 'lead_time': 90,
 'arrival_year': 2018, 'arrival_month': 10, 'arrival_date': 12,
 'market_segment_type': 'Online', 'repeated_guest': 0,
 'no_of_previous_cancellations': 0, 'no_of_previous_bookings_not_canceled': 0,
 'avg_price_per_room': 110.0, 'no_of_special_requests': 1
}

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_prediction():
    r = client.post('/predict', json=EXAMPLE)
    assert r.status_code == 200, r.text
    assert 0 <= r.json()['cancel_probability'] <= 1
    assert r.json()['prediction'] in ('Canceled', 'Not_Canceled')

def test_bad_input():
    payload = dict(EXAMPLE, lead_time=-5)
    assert client.post('/predict', json=payload).status_code == 422

def test_metrics():
    r = client.get('/metrics')
    assert r.status_code == 200
    assert 'hotel_predictions_total' in r.text
