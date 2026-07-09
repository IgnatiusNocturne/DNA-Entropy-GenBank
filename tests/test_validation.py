"""Tests for sequence normalization and validation (docs/DESIGN.md §5)."""

from __future__ import annotations

import pytest

from dna_entropy.validation import (
    ValidatedSequence,
    ValidationError,
    validate_sequence,
)


def test_plain_valid_sequence() -> None:
    result = validate_sequence("ATGCATGCAT")
    assert isinstance(result, ValidatedSequence)
    assert result.seq == "ATGCATGCAT"


def test_lowercase_is_uppercased() -> None:
    assert validate_sequence("atgcatgcat").seq == "ATGCATGCAT"


def test_whitespace_and_newlines_removed() -> None:
    assert validate_sequence("AT GC\nAC GT\tAA\r\nCC").seq == "ATGCACGTAACC"


def test_digits_removed_with_notice() -> None:
    result = validate_sequence("1 ATGCATGC 60\n61 ACGTACGT 120")
    assert result.seq == "ATGCATGCACGTACGT"
    assert any("digit" in n.lower() for n in result.notices)


def test_fasta_header_stripped_with_notice() -> None:
    result = validate_sequence(">seq1 some description\nATGCATGCAT")
    assert result.seq == "ATGCATGCAT"
    assert any("header" in n.lower() for n in result.notices)


def test_blank_lines_before_header_ok() -> None:
    result = validate_sequence("\n\n>hdr\nATGCATGCAT")
    assert result.seq == "ATGCATGCAT"


def test_invalid_char_reports_first_position() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_sequence("ATGBCATGCA")  # 'B' at index 3 -> position 4
    msg = str(exc.value)
    assert "position 4" in msg
    assert "B" in msg


def test_invalid_char_reports_total_count() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_sequence("ATXGCXTGCX")  # three 'X'
    assert "3 non-ACGT" in str(exc.value)


def test_ambiguity_code_gives_hint() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_sequence("ATGCNATGCA")  # 'N' is IUPAC ambiguity
    assert "ambiguity" in str(exc.value).lower()


def test_rna_without_flag_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_sequence("AUGCAUGCAU")
    assert "rna" in str(exc.value).lower()


def test_rna_with_flag_is_converted() -> None:
    result = validate_sequence("AUGCAUGCAU", rna=True)
    assert result.seq == "ATGCATGCAT"
    assert any("U->T" in n for n in result.notices)


def test_empty_after_cleaning_is_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_sequence("   \n  123  \n  ")


def test_header_only_is_rejected_as_empty() -> None:
    with pytest.raises(ValidationError):
        validate_sequence(">only a header\n")


def test_over_max_len_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_sequence("A" * 100, max_len=50)
    assert "exceeds" in str(exc.value)


def test_short_sequence_warns_but_passes() -> None:
    result = validate_sequence("ATG", min_len=10)
    assert result.seq == "ATG"
    assert any("short" in n.lower() for n in result.notices)


def test_clean_sequence_has_no_notices() -> None:
    assert validate_sequence("ATGCATGCATGC").notices == []
