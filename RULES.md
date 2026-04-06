# RULES.md

**MOST IMPORTANT** - Solve problems from the root, don't put patches to deal with the symptoms.

## Code Standards
- Read every file before modifying it. Make minimal changes — no unasked extras.
- No security vulnerabilities: no injected secrets, no exposed PII, no unsafe evals.

## Documentation
- Every new file gets a top-level description comment. Every function gets a docstring (purpose, args, return).
- add a # comment above lines that may be unclear.
- Update and maintain a professional README.md file.


## Design
- **Single Responsibility**: if you need "and" to describe what a function does, split it.
- **Fail fast**: raise errors early with clear messages. Never silently swallow exceptions or return `None` where an error is appropriate.
- **Separation of concerns**: keep I/O (API calls, file reads) out of business logic.
- **Dependency injection**: pass dependencies in — don't reach for globals or singletons inside logic.

## Error Handling
- Use specific exception types. Never bare `except:` unless re-raising.
- Always include context in error messages: what failed, what the input was, what was expected.

## Testing
- One behavior per test. Name tests `test_<scenario>_<expected_outcome>`.
- Mock only at system boundaries (external APIs, file system). Never mock internals.

## Security
- Never log sensitive data (PII, tokens, medical content).
- Secrets via environment variables only — never hard-coded.

## Self-Review (before every response with code)
1. Did I read every file I modified?
2. Is the change minimal — no unasked extras?
3. Any security issues or exposed secrets?
4. Is the code correct and would it actually run?
5. Is there a simpler way?