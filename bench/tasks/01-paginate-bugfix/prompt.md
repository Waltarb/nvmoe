# Bug: pagination is off by one

`paginate()` is used by our product list. Pages are **1-based** (page 1 is the first page), but the current code treats them as 0-based, so page 1 skips the first items and `hasNext` is wrong on the last page.

Expected behaviour:
- `page` is 1-based.
- `page` below 1 is clamped to 1, above `totalPages` is clamped to `totalPages`. The returned `page` is the clamped value.
- `totalPages` is at least 1, also for an empty list.
- `hasNext` / `hasPrev` reflect the clamped page.
