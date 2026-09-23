export interface Page<T> {
  items: T[];
  page: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export function paginate<T>(items: T[], page: number, pageSize: number): Page<T> {
  const totalPages = Math.ceil(items.length / pageSize);
  const start = page * pageSize;
  const end = start + pageSize;
  return {
    items: items.slice(start, end),
    page,
    totalPages,
    hasNext: page < totalPages,
    hasPrev: page > 0,
  };
}
