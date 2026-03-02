from __future__ import annotations

import pytest

from runpodx.models.pod import PodModel
from runpodx.resources.pods import Pod, PodManager


class TestPodModel:
    def test_parse_api_response(self, sample_pod_data):
        model = PodModel.model_validate(sample_pod_data)
        assert model.id == "pod_abc123"
        assert model.name == "test-pod"
        assert model.desired_status == "RUNNING"
        assert model.gpu_type_id == "NVIDIA GeForce RTX 4090"
        assert model.cost_per_hr == 0.44
        assert model.public_ip == "1.2.3.4"

    def test_status_properties(self):
        running = PodModel(id="1", desired_status="RUNNING")
        assert running.is_running is True
        assert running.is_stopped is False

        stopped = PodModel(id="2", desired_status="EXITED")
        assert stopped.is_stopped is True
        assert stopped.is_running is False

        terminated = PodModel(id="3", desired_status="TERMINATED")
        assert terminated.is_terminated is True

    def test_extra_fields_allowed(self):
        data = {"id": "1", "unknownField": "value"}
        model = PodModel.model_validate(data)
        assert model.id == "1"


class TestPod:
    def test_attribute_delegation(self, mock_transport, sample_pod_data):
        transport = mock_transport()
        model = PodModel.model_validate(sample_pod_data)
        pod = Pod(model, transport)
        assert pod.name == "test-pod"
        assert pod.id == "pod_abc123"
        assert pod.cost_per_hr == 0.44

    def test_stop(self, mock_transport, sample_pod_data):
        stopped_data = {**sample_pod_data, "desiredStatus": "EXITED"}
        transport = mock_transport(
            responses={
                "POST /pods/pod_abc123/stop": {},
                "GET /pods/pod_abc123": stopped_data,
            }
        )
        pod = Pod(PodModel.model_validate(sample_pod_data), transport)
        pod.stop()
        assert pod.desired_status == "EXITED"
        assert len(transport.calls) == 2

    def test_start(self, mock_transport, sample_pod_data):
        transport = mock_transport(
            responses={
                "POST /pods/pod_abc123/start": {},
                "GET /pods/pod_abc123": sample_pod_data,
            }
        )
        pod = Pod(PodModel.model_validate(sample_pod_data), transport)
        pod.start()
        assert len(transport.calls) == 2

    def test_delete(self, mock_transport, sample_pod_data):
        transport = mock_transport(
            responses={"DELETE /pods/pod_abc123": {}}
        )
        pod = Pod(PodModel.model_validate(sample_pod_data), transport)
        pod.delete()
        assert transport.calls[0] == ("DELETE", "/pods/pod_abc123", {})

    def test_repr(self, mock_transport, sample_pod_data):
        pod = Pod(PodModel.model_validate(sample_pod_data), mock_transport())
        assert "pod_abc123" in repr(pod)
        assert "test-pod" in repr(pod)
        assert "RUNNING" in repr(pod)


class TestPodManager:
    def test_list(self, mock_transport, sample_pod_data):
        transport = mock_transport(
            responses={"GET /pods": [sample_pod_data, sample_pod_data]}
        )
        manager = PodManager(transport)
        pods = manager.list()
        assert len(pods) == 2
        assert all(isinstance(p, Pod) for p in pods)

    def test_get(self, mock_transport, sample_pod_data):
        transport = mock_transport(
            responses={"GET /pods/pod_abc123": sample_pod_data}
        )
        manager = PodManager(transport)
        pod = manager.get("pod_abc123")
        assert pod.id == "pod_abc123"

    def test_create_from_dict(self, mock_transport, sample_pod_data):
        transport = mock_transport(
            responses={"POST /pods": sample_pod_data}
        )
        manager = PodManager(transport)
        pod = manager.create({"image": "pytorch/pytorch:2.1.0"})
        assert pod.id == "pod_abc123"
        call = transport.calls[0]
        assert call[0] == "POST"
        assert call[1] == "/pods"
        assert call[2]["json"]["imageName"] == "pytorch/pytorch:2.1.0"

    def test_delete(self, mock_transport):
        transport = mock_transport(
            responses={"DELETE /pods/pod_abc123": {}}
        )
        manager = PodManager(transport)
        manager.delete("pod_abc123")
        assert transport.calls[0] == ("DELETE", "/pods/pod_abc123", {})
