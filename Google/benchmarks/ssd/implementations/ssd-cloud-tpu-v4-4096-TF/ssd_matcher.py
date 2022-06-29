"""Matcher for MLPerf 2.0 SSD submission."""

import tensorflow.compat.v1 as tf

from object_detection import argmax_matcher
from object_detection import shape_utils


class Matcher(argmax_matcher.ArgMaxMatcher):
  """Matcher for MLPerf 2.0.

  The main difference between this class and ArgMaxMatcher is
  allow_low_quality_matches. See below for details.
  """

  def __init__(self, allow_low_quality_matches=False, tol=1e-4, **kwargs):
    """Construct a matcher for MLPerf 2.0 SSD.

    Args:
      allow_low_quality_matches: See below code for details.
      tol: Tolerance for checking IoU equailty.
      **kwargs: Passed to ArgMaxMatcher
    """
    if kwargs.get('force_match_for_each_row',
                  False) and allow_low_quality_matches:
      raise ValueError(
          'force_match_for_each_row and allow_low_quality_matches cannot be true at the same time.'
      )
    super().__init__(**kwargs)
    self._allow_low_quality_matches = allow_low_quality_matches
    self._tol = tol

  def _match(self, similarity_matrix):
    """Copied from ArgMaxMatcher with changes.

    Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: tensor of shape [N, M] representing any similarity
        metric.

    Returns:
      Match object with corresponding matches for each of M columns.
    """

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
          similarity_matrix)
      return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      # Matches for each column
      matches = tf.argmax(similarity_matrix, 0, output_type=tf.int32)
      raw_matches = matches

      # Deal with matched and unmatched threshold
      if self._matched_threshold is not None:
        # Get logical indices of ignored and unmatched columns as tf.int64
        matched_vals = tf.reduce_max(similarity_matrix, 0)
        below_unmatched_threshold = tf.greater(self._unmatched_threshold,
                                               matched_vals)
        between_thresholds = tf.logical_and(
            tf.greater_equal(matched_vals, self._unmatched_threshold),
            tf.greater(self._matched_threshold, matched_vals))

        if self._negatives_lower_than_unmatched:
          matches = self._set_values_using_indicator(matches,
                                                     below_unmatched_threshold,
                                                     -1)
          matches = self._set_values_using_indicator(matches,
                                                     between_thresholds, -2)
        else:
          matches = self._set_values_using_indicator(matches,
                                                     below_unmatched_threshold,
                                                     -2)
          matches = self._set_values_using_indicator(matches,
                                                     between_thresholds, -1)
      if self._force_match_for_each_row:
        similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
            similarity_matrix)
        force_match_column_ids = tf.argmax(
            similarity_matrix, 1, output_type=tf.int32)
        force_match_column_indicators = tf.one_hot(
            force_match_column_ids, depth=similarity_matrix_shape[1])
        force_match_row_ids = tf.argmax(
            force_match_column_indicators, 0, output_type=tf.int32)
        force_match_column_mask = tf.cast(
            tf.reduce_max(force_match_column_indicators, 0), tf.bool)
        final_matches = tf.where(force_match_column_mask, force_match_row_ids,
                                 matches)
        return final_matches
      elif self._allow_low_quality_matches:
        # (Copied from PyT reference)
        # https://github.com/mlcommons/training/blob/a0671ab8c9668d4acf86977ad6c9d36995431197/single_stage_detector/ssd/model/utils.py#L313
        # Produce additional matches for predictions that have only low-quality
        # matches. Specifically, for each ground-truth find the set of
        # predictions that have maximum overlap with it (including ties); for
        # each prediction in that set, if it is unmatched, then match it to the
        # ground-truth with which it has the highest quality value.
        similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
            similarity_matrix)
        highest_sim_per_row = tf.reduce_max(
            similarity_matrix, axis=1, keepdims=True)
        # 1 means this column is this row's highest sim match. Ties are kept.
        # Use a small `tol` to tolerate floating point error
        force_match_column_indicators = tf.greater_equal(
            similarity_matrix, highest_sim_per_row - self._tol)

        # 1 means this column is some row's highest sim match
        force_match_clumn_mask = tf.reduce_any(
            force_match_column_indicators, axis=0)
        final_matches = tf.where_v2(force_match_clumn_mask, raw_matches,
                                    matches)
        return final_matches
      else:
        return matches

    if similarity_matrix.shape.is_fully_defined():
      if similarity_matrix.shape[0].value == 0:
        return _match_when_rows_are_empty()
      else:
        return _match_when_rows_are_non_empty()
    else:
      return tf.cond(
          tf.greater(tf.shape(similarity_matrix)[0], 0),
          _match_when_rows_are_non_empty, _match_when_rows_are_empty)
