from __future__ import annotations

from runpodx.training.scripts.grpo_profile_ascii_art import (
    anti_loop_reward,
    build_reward_functions,
    soft_overlong_penalty,
    target_length_reward,
    target_similarity_reward,
)


def test_build_reward_functions_count():
    reward_funcs = build_reward_functions({})
    assert len(reward_funcs) == 4


def test_target_similarity_prefers_matching_shape():
    target = ["/\\\n||\n\\/"]
    matching = ["/\\\n||\n\\/"]
    mismatched = [".---.\n$--$--$--"]

    matching_score = target_similarity_reward(matching, target=target)[0]
    mismatched_score = target_similarity_reward(mismatched, target=target)[0]

    assert matching_score > mismatched_score


def test_target_length_penalizes_short_degenerate_outputs():
    target = ["/\\\n||\n||\n||\n\\/"]
    short = [".---."]
    closer = ["/\\\n||\n||\n\\/"]

    short_score = target_length_reward(short, target=target)[0]
    closer_score = target_length_reward(closer, target=target)[0]

    assert closer_score > short_score


def test_target_length_penalizes_near_cap_rollouts():
    target = ["/\\\n||\n||\n\\/"]
    completion = ["/\\\n||\n||\n\\/"]

    normal_score = target_length_reward(
        completion,
        target=target,
        completion_ids=[[1] * 64],
        max_completion_length=192,
        soft_overlong_cache=32,
    )[0]
    near_cap_score = target_length_reward(
        completion,
        target=target,
        completion_ids=[[1] * 191],
        max_completion_length=192,
        soft_overlong_cache=32,
    )[0]

    assert normal_score > near_cap_score


def test_anti_loop_penalizes_degenerate_ngram_runs():
    clean = ["/\\\n||\n\\/"]
    degenerate = [".---.\n$--$--$--$--$--$--$--$--"]

    clean_score = anti_loop_reward(clean)[0]
    degenerate_score = anti_loop_reward(degenerate)[0]

    assert clean_score > degenerate_score


def test_soft_overlong_penalty_scales_near_cap():
    assert soft_overlong_penalty([1] * 100, max_completion_length=192, soft_overlong_cache=32) == 0
    assert soft_overlong_penalty([1] * 191, max_completion_length=192, soft_overlong_cache=32) < 0
