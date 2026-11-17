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
        running = PodModel(id="1", status="RUNNING")
        assert running.is_running is True
        assert running.is_stopped is False

        stopped = PodModel(id="2", status="EXITED")
        assert stopped.is_stopped is True
        assert stopped.is_running is False

        terminated = PodModel(id="3", status="TERMINATED")
        assert terminated.is_terminated is True

        failed = PodModel(id="4", status="ERROR")
        assert failed.is_error is True

    def test_v1_names_and_dump(self, sample_pod_data):
        model = PodModel.model_validate(sample_pod_data)
        assert model.port_mappings == {"22": 15538, "8888": 18888}
        assert model.gpu_count == 1
        assert model.image_name == "runpod/pytorch:2.1.0"
        dumped = model.model_dump(by_alias=True)
        assert dumped["publicIp"] == "1.2.3.4"
        assert dumped["portMappings"]["22"] == 15538

    def test_public_ip_from_ssh_direct_alone(self):
        model = PodModel.model_validate({"id": "1", "ssh": {"direct": {"host": "5.6.7.8", "port": 2201}}})
        assert model.public_ip == "5.6.7.8"
        assert model.port_mappings == {"22": 2201}

    def test_not_ready(self):
        model = PodModel.model_validate({"id": "1", "status": "STARTING", "runtime": None})
        assert model.public_ip is None
        assert not model.is_running

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
        stopped_data = {**sample_pod_data, "status": "EXITED"}
        transport = mock_transport(
            responses={
                "POST /pods/pod_abc123/action": {},
                "GET /pods/pod_abc123": stopped_data,
            }
        )
        pod = Pod(PodModel.model_validate(sample_pod_data), transport)
        pod.stop()
        assert pod.desired_status == "EXITED"
        assert transport.calls[0] == ("POST", "/pods/pod_abc123/action", {"json": {"action": "stop"}})
        assert len(transport.calls) == 2

    def test_start(self, mock_transport, sample_pod_data):
        transport = mock_transport(
            responses={
                "POST /pods/pod_abc123/action": {},
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
            responses={"GET /pods": {"pods": [sample_pod_data, sample_pod_data],
                                     "pagination": {"nextCursor": None, "hasNextPage": False}}}
        )
        manager = PodManager(transport)
        pods = manager.list()
        assert len(pods) == 2
        assert all(isinstance(p, Pod) for p in pods)
        assert manager.list(name="nobody") == []

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
        assert call[2]["json"]["image"] == "pytorch/pytorch:2.1.0"

    def test_create_tries_each_gpu_type(self, sample_pod_data):
        from runpodx.http.errors import ServerError

        class Busy:
            def __init__(self):
                self.calls = []

            def request(self, method, path, **kw):
                self.calls.append(kw["json"]["gpu"]["id"])
                if kw["json"]["gpu"]["id"] == "A40":
                    raise ServerError("no instances currently available", status_code=500)
                return sample_pod_data
        t = Busy()
        pod = PodManager(t).create({"image": "x", "gpu": {"type_ids": ["A40", "L40S"]}})
        assert t.calls == ["A40", "L40S"]
        assert pod.id == "pod_abc123"

    def test_delete(self, mock_transport):
        transport = mock_transport(
            responses={"DELETE /pods/pod_abc123": {}}
        )
        manager = PodManager(transport)
        manager.delete("pod_abc123")
        assert transport.calls[0] == ("DELETE", "/pods/pod_abc123", {})
