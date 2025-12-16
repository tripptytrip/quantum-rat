
import json
from agents.dna import AgentDNA

def test_agentdna_serialization():
    """
    Tests that AgentDNA can be serialized to JSON and back.
    """
    dna = AgentDNA(
        agent_id="test_agent",
        version="1.0",
        params={"param1": 1.0, "param2": "value2"}
    )
    
    json_str = dna.to_json()
    
    # Check that it's a valid json
    data = json.loads(json_str)
    assert data["agent_id"] == "test_agent"
    
    # Check deserialization
    new_dna = AgentDNA.from_json(json_str)
    assert new_dna == dna

def test_agentdna_fingerprint_stability():
    """
    Tests that the fingerprint of an AgentDNA instance is stable.
    """
    dna1 = AgentDNA(
        agent_id="test_agent",
        version="1.0",
        params={"param1": 1.0, "param2": "value2"}
    )
    
    dna2 = AgentDNA(
        agent_id="test_agent",
        version="1.0",
        params={"param2": "value2", "param1": 1.0} # Note reversed order
    )
    
    assert dna1.fingerprint() == dna2.fingerprint()

def test_agentdna_fingerprint_uniqueness():
    """
    Tests that different AgentDNA instances have different fingerprints.
    """
    dna1 = AgentDNA(
        agent_id="test_agent_1",
        version="1.0",
        params={"param1": 1.0}
    )
    
    dna2 = AgentDNA(
        agent_id="test_agent_2",
        version="1.0",
        params={"param1": 1.0}
    )
    
    dna3 = AgentDNA(
        agent_id="test_agent_1",
        version="1.1",
        params={"param1": 1.0}
    )

    dna4 = AgentDNA(
        agent_id="test_agent_1",
        version="1.0",
        params={"param1": 2.0}
    )
    
    assert dna1.fingerprint() != dna2.fingerprint()
    assert dna1.fingerprint() != dna3.fingerprint()
    assert dna1.fingerprint() != dna4.fingerprint()

