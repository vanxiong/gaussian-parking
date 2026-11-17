import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import install_surroundocc_inputs as assets
from surroundocc_protocol import confined

@pytest.fixture
def confined_copy(monkeypatch,tmp_path):
    monkeypatch.setattr(assets,'confined',lambda path:confined(path,tmp_path))
    return tmp_path

def test_archive_member_install_is_complete(confined_copy):
    target=confined_copy/'labels/test.npy'
    assert assets.copy_member(io.BytesIO(b'abc'),target,3)
    assert target.read_bytes()==b'abc'
    assert not assets.copy_member(io.BytesIO(b'abc'),target,3)

def test_existing_mismatched_file_is_preserved(confined_copy):
    target=confined_copy/'original';target.write_bytes(b'original')
    with pytest.raises(ValueError):assets.copy_member(io.BytesIO(b'abc'),target,3)
    assert target.read_bytes()==b'original'

def test_truncated_archive_member_never_becomes_final(confined_copy):
    target=confined_copy/'truncated.npy'
    with pytest.raises(ValueError):assets.copy_member(io.BytesIO(b'a'),target,3)
    assert not target.exists()

def test_archive_path_cannot_escape_data_root(confined_copy):
    with pytest.raises(ValueError):assets.copy_member(io.BytesIO(b'a'),confined_copy/'../escape',1)
