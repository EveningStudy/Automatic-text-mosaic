import unittest
import json
from app import app  # Import your Flask app from the main script

class MosaicTextRegionTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

        # Setup parameters for testing mosaic_text_region endpoint
        self.test_data = {
            "x": 50,
            "y": 100,
            "filename": "",
            "mosaic_type": "GaussianBlur",
            "displayed_width": 500,
            "displayed_height": 400
        }

    def test_mosaic_text_region_success(self):
        """Test that mosaic_text_region applies mosaic and returns success."""

        # Convert the test data to JSON format for the request
        response = self.client.post(
            '/mosaic',  # Endpoint for manual mosaic
            data=json.dumps(self.test_data),
            content_type='application/json'
        )

        # Assert the response to check if the mosaic was applied successfully
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('success', json_response)
        self.assertTrue(json_response['success'])

    def test_mosaic_text_region_invalid_coordinates(self):
        """Test mosaic_text_region with out-of-bound coordinates."""

        # Modify coordinates to be outside typical bounds for testing
        out_of_bounds_data = self.test_data.copy()
        out_of_bounds_data.update({"x": 10000, "y": 10000})

        response = self.client.post(
            '/mosaic',
            data=json.dumps(out_of_bounds_data),
            content_type='application/json'
        )

        # Assert the response should still return success but without mosaic applied
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('success', json_response)
        self.assertTrue(json_response['success'])



if __name__ == '__main__':
    unittest.main()