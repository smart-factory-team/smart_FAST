import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

# Import the router and dependencies
from services.painting_process_equipment_defect_detection_model_service.app.routers.model_info import router
from services.painting_process_equipment_defect_detection_model_service.app.dependencies import get_config


# Create a test app with the router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestGetModelInfo:
    """Test cases for the get_model_info endpoint - Using pytest testing framework"""
    
    def test_get_model_info_success_complete_config(self):
        """Test successful retrieval of model info with complete configuration"""
        # Arrange
        mock_config = {
            "model": {
                "name": "defect_detection_v1",
                "version": "1.0.0",
                "architecture": "ResNet50",
                "model_repo_id": "test/model",
                "model_filename": "model.pkl"
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "threshold": {
                "confidence": 0.8,
                "iou": 0.5
            },
            "features": {
                "input_size": [224, 224, 3],
                "num_classes": 5,
                "preprocessing": "normalize"
            },
            "other_config": "should_be_filtered_out"
        }
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify only expected keys are returned
        expected_keys = {"model", "training", "threshold", "features"}
        assert set(data.keys()) == expected_keys
        
        # Verify the data matches expected structure
        assert data["model"] == mock_config["model"]
        assert data["training"] == mock_config["training"]
        assert data["threshold"] == mock_config["threshold"]
        assert data["features"] == mock_config["features"]
        
        # Verify filtered config is not included
        assert "other_config" not in data

    def test_get_model_info_partial_config(self):
        """Test model info retrieval with partial configuration"""
        # Arrange
        mock_config = {
            "model": {"name": "test_model"},
            "training": {"epochs": 50}
            # Missing threshold and features
        }
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["model"] == {"name": "test_model"}
        assert data["training"] == {"epochs": 50}
        assert data["threshold"] is None
        assert data["features"] is None

    def test_get_model_info_empty_config(self):
        """Test model info retrieval with empty configuration"""
        # Arrange
        mock_config = {}
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # All values should be None for missing keys
        assert data["model"] is None
        assert data["training"] is None
        assert data["threshold"] is None
        assert data["features"] is None

    def test_get_model_info_none_values_in_config(self):
        """Test model info retrieval when config contains None values"""
        # Arrange
        mock_config = {
            "model": None,
            "training": None,
            "threshold": {"confidence": 0.9},
            "features": None
        }
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["model"] is None
        assert data["training"] is None
        assert data["threshold"] == {"confidence": 0.9}
        assert data["features"] is None

    def test_get_model_info_nested_complex_config(self):
        """Test model info retrieval with deeply nested configuration"""
        # Arrange
        mock_config = {
            "model": {
                "name": "advanced_model",
                "layers": {
                    "conv1": {"filters": 32, "kernel_size": 3},
                    "conv2": {"filters": 64, "kernel_size": 3}
                },
                "optimizer": {
                    "type": "Adam",
                    "parameters": {"lr": 0.001, "beta1": 0.9}
                }
            },
            "training": {
                "data": {
                    "train_split": 0.8,
                    "val_split": 0.2,
                    "augmentation": ["rotate", "flip", "scale"]
                },
                "callbacks": {
                    "early_stopping": {"patience": 10},
                    "model_checkpoint": {"save_best_only": True}
                }
            },
            "threshold": {
                "detection": {"min": 0.5, "max": 0.95},
                "classification": {"confidence": 0.8}
            },
            "features": {
                "extraction": {
                    "method": "CNN",
                    "layers": ["conv", "pool", "dense"]
                },
                "preprocessing": {
                    "normalization": "z-score",
                    "resize": [224, 224]
                }
            }
        }
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify deep nested structures are preserved
        assert data["model"]["layers"]["conv1"]["filters"] == 32
        assert data["training"]["data"]["augmentation"] == ["rotate", "flip", "scale"]
        assert data["threshold"]["detection"]["min"] == 0.5
        assert data["features"]["preprocessing"]["normalization"] == "z-score"

    def test_get_model_info_config_dependency_injection(self):
        """Test that the dependency injection works correctly"""
        # This test verifies that get_config is properly called as a dependency
        
        mock_config = {"model": {"test": "dependency_injection"}}
        
        def mock_get_config():
            return mock_config
        
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = mock_get_config
            response = client.get("/")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["model"] == {"test": "dependency_injection"}

    def test_get_model_info_response_structure_consistency(self):
        """Test that response always has the same structure regardless of input"""
        test_configs = [
            {},  # Empty
            {"model": "test"},  # Partial
            {  # Complete
                "model": {"name": "test"},
                "training": {"epochs": 10},
                "threshold": {"confidence": 0.8},
                "features": {"size": 224}
            }
        ]
        
        for i, config in enumerate(test_configs):
            with app.dependency_overrides:
                app.dependency_overrides[get_config] = lambda c=config: c
                response = client.get("/")
                
                assert response.status_code == 200
                data = response.json()
                
                # Always has the same keys
                expected_keys = {"model", "training", "threshold", "features"}
                assert set(data.keys()) == expected_keys, f"Failed for config {i}: {config}"

    def test_get_model_info_large_config_data(self):
        """Test handling of large configuration data"""
        # Arrange - Create a large config with many nested elements
        large_config = {
            "model": {
                "layers": {f"layer_{i}": {"neurons": i * 10} for i in range(100)},
                "weights": list(range(1000))
            },
            "training": {
                "history": {f"epoch_{i}": {"loss": i * 0.01} for i in range(500)},
                "data_points": list(range(10000))
            },
            "threshold": {
                "class_thresholds": {f"class_{i}": 0.1 + i * 0.01 for i in range(50)}
            },
            "features": {
                "feature_map": {f"feature_{i}": f"description_{i}" for i in range(200)}
            }
        }
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: large_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify large data is handled correctly
        assert len(data["model"]["layers"]) == 100
        assert len(data["training"]["history"]) == 500
        assert len(data["threshold"]["class_thresholds"]) == 50
        assert len(data["features"]["feature_map"]) == 200

    def test_get_model_info_config_with_special_characters(self):
        """Test handling of configuration with special characters and unicode"""
        # Arrange
        mock_config = {
            "model": {
                "name": "모델_이름_한글",
                "description": "Model with émojis 🤖 and special chars: @#$%^&*()",
                "path": "/path/with spaces/and-dashes/model.pkl"
            },
            "training": {
                "dataset": "데이터셋_2023",
                "notes": "Training notes with newlines\nand tabs\t and quotes \"quotes\""
            },
            "threshold": {
                "μ_threshold": 0.8,  # Greek letter mu
                "σ_threshold": 0.2   # Greek letter sigma
            },
            "features": {
                "encoding": "utf-8",
                "special_features": ["特殊文字", "спеціальні символи", "الأحرف الخاصة"]
            }
        }
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify special characters are preserved
        assert data["model"]["name"] == "모델_이름_한글"
        assert "🤖" in data["model"]["description"]
        assert data["threshold"]["μ_threshold"] == 0.8
        assert "特殊文字" in data["features"]["special_features"]

    def test_get_model_info_realistic_ai_model_config(self):
        """Test with realistic AI model configuration for painting defect detection"""
        # Arrange - Realistic configuration for a defect detection model
        realistic_config = {
            "model": {
                "name": "painting_defect_detector",
                "version": "2.1.0",
                "architecture": "YOLOv8",
                "model_repo_id": "company/painting-defect-model",
                "model_filename": "painting_defect_v2_1_0.pkl",
                "input_size": [640, 640, 3],
                "num_classes": 7,
                "class_names": [
                    "scratch", "dent", "bubble", "color_mismatch",
                    "contamination", "texture_defect", "edge_defect"
                ]
            },
            "training": {
                "dataset_version": "v3.2",
                "total_images": 15000,
                "train_images": 12000,
                "val_images": 2000,
                "test_images": 1000,
                "epochs": 150,
                "batch_size": 16,
                "learning_rate": 0.0001,
                "optimizer": "AdamW",
                "augmentations": [
                    "random_rotation", "brightness_adjustment",
                    "contrast_enhancement", "noise_injection"
                ],
                "early_stopping_patience": 15,
                "best_val_accuracy": 0.934,
                "training_time_hours": 48.5
            },
            "threshold": {
                "confidence_threshold": 0.75,
                "nms_threshold": 0.45,
                "min_detection_size": 50,
                "class_specific_thresholds": {
                    "scratch": 0.8,
                    "dent": 0.75,
                    "bubble": 0.85,
                    "color_mismatch": 0.7,
                    "contamination": 0.8,
                    "texture_defect": 0.75,
                    "edge_defect": 0.8
                }
            },
            "features": {
                "backbone": "CSPDarknet53",
                "neck": "PANet",
                "head": "YOLOv8Head",
                "feature_pyramid_levels": [3, 4, 5],
                "anchor_sizes": [[10, 13], [16, 30], [33, 23]],
                "preprocessing": {
                    "normalization": "imagenet",
                    "resize_method": "letterbox",
                    "padding_color": [114, 114, 114]
                },
                "postprocessing": {
                    "nms_method": "batched_nms",
                    "max_detections_per_image": 100,
                    "score_threshold": 0.1
                }
            },
            "deployment": {
                "inference_device": "cuda:0",
                "batch_inference": True,
                "max_batch_size": 8,
                "optimization": "tensorrt"
            }
        }
        
        # Act
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: realistic_config
            response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify expected sections are returned
        assert "model" in data
        assert "training" in data
        assert "threshold" in data
        assert "features" in data
        
        # Verify deployment section is filtered out (not in the expected keys)
        assert "deployment" not in data
        
        # Verify specific realistic values
        assert data["model"]["name"] == "painting_defect_detector"
        assert data["model"]["num_classes"] == 7
        assert len(data["model"]["class_names"]) == 7
        
        assert data["training"]["total_images"] == 15000
        assert data["training"]["best_val_accuracy"] == 0.934
        
        assert data["threshold"]["confidence_threshold"] == 0.75
        assert len(data["threshold"]["class_specific_thresholds"]) == 7
        
        assert data["features"]["backbone"] == "CSPDarknet53"
        assert data["features"]["preprocessing"]["normalization"] == "imagenet"


class TestGetModelInfoEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_get_model_info_config_not_loaded_error(self):
        """Test behavior when get_config raises HTTPException for config not loaded"""
        
        def mock_get_config_error():
            raise HTTPException(status_code=500, detail="Configuration not loaded")
        
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = mock_get_config_error
            response = client.get("/")
            
            # FastAPI should return the HTTP exception as response
            assert response.status_code == 500
            assert response.json()["detail"] == "Configuration not loaded"

    def test_get_model_info_config_returns_non_dict(self):
        """Test behavior when get_config returns non-dictionary type"""
        
        # Test with various non-dict types
        non_dict_values = [None, "string", 123, [], True]
        
        for value in non_dict_values:
            def mock_get_config_non_dict(v=value):
                return v
            
            with app.dependency_overrides:
                app.dependency_overrides[get_config] = mock_get_config_non_dict
                response = client.get("/")
                
                # Should return 500 internal server error due to AttributeError
                assert response.status_code == 500

    def test_get_model_info_config_with_unexpected_data_types(self):
        """Test handling of configuration with unexpected data types"""
        # Arrange
        mock_config = {
            "model": {
                "complex_number": complex(1, 2),
                "set_data": {1, 2, 3},  # Sets are not JSON serializable
                "function": lambda x: x,  # Functions are not serializable
            },
            "training": {
                "normal_data": "should work"
            },
            "threshold": None,
            "features": []
        }
        
        # Act & Assert
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
            
            # Should return 500 due to JSON serialization error
            assert response.status_code == 500


class TestGetModelInfoDocumentation:
    """Test API documentation and metadata"""
    
    def test_endpoint_has_correct_docstring(self):
        """Verify the endpoint has proper Korean documentation"""
        from services.painting_process_equipment_defect_detection_model_service.app.routers.model_info import get_model_info
        
        assert get_model_info.__doc__ is not None
        assert "로드된 AI 모델의 설정 정보를 반환합니다" in get_model_info.__doc__
    
    def test_endpoint_route_configuration(self):
        """Test that the route is configured correctly"""
        # Test with a valid config to ensure route exists
        mock_config = {"model": None, "training": None, "threshold": None, "features": None}
        
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
            
            # Should not be 404 (route exists)
            assert response.status_code != 404
            # Should be 200 with valid response
            assert response.status_code == 200

    def test_endpoint_response_content_type(self):
        """Test that the endpoint returns correct content type"""
        mock_config = {"model": {"test": "json"}}
        
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/")
            
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]

    def test_endpoint_handles_get_method_only(self):
        """Test that endpoint only accepts GET method"""
        mock_config = {"model": None}
        
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            
            # GET should work
            get_response = client.get("/")
            assert get_response.status_code == 200
            
            # Other methods should not be allowed
            post_response = client.post("/")
            assert post_response.status_code == 405  # Method Not Allowed
            
            put_response = client.put("/")
            assert put_response.status_code == 405  # Method Not Allowed
            
            delete_response = client.delete("/")
            assert delete_response.status_code == 405  # Method Not Allowed


class TestGetModelInfoPerformance:
    """Test performance and concurrency aspects"""
    
    def test_get_model_info_concurrent_requests(self):
        """Test multiple concurrent requests to the endpoint"""
        import threading
        import time
        
        mock_config = {
            "model": {"concurrent_test": True},
            "training": {"epochs": 100},
            "threshold": {"confidence": 0.8},
            "features": {"size": 224}
        }
        
        results = []
        errors = []
        
        def make_request():
            try:
                with app.dependency_overrides:
                    app.dependency_overrides[get_config] = lambda: mock_config
                    response = client.get("/")
                    results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads to test concurrency
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Assert all requests succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for status in results)
        assert len(results) == 10
        
        # Performance check - should complete within reasonable time
        assert end_time - start_time < 5.0  # Should complete within 5 seconds

    def test_get_model_info_response_time(self):
        """Test that endpoint responds within acceptable time"""
        import time
        
        mock_config = {
            "model": {"performance_test": True},
            "training": {"data": list(range(1000))},  # Moderate size data
            "threshold": {"values": list(range(100))},
            "features": {"map": {f"key_{i}": f"value_{i}" for i in range(500)}}
        }
        
        with app.dependency_overrides:
            app.dependency_overrides[get_config] = lambda: mock_config
            
            start_time = time.time()
            response = client.get("/")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            # Response should be fast (under 1 second for moderate data)
            assert response_time < 1.0


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__])