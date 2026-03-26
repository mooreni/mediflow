# RULES.md

Rules for Claude Code when working in this repository.

## Coding Standards

- Follow SOLID, DRY, KISS. Make minimal changes — no unasked extras, no over-engineering.
- Read every file before modifying it. Validate only at system boundaries (user input, external APIs).
- Never introduce security vulnerabilities (injection, XSS, exposed secrets, unsafe evals).
- Verify code is correct and working before presenting it.
- **Docs**: Every new file gets a top-level description comment. Every function gets a docstring: purpose, args, and return value.

## Design Principles

- **Single Responsibility**: Each module, class, and function does one thing. If you need "and" to describe what it does, split it.
- **Fail fast**: Raise errors early with clear messages. Never silently swallow exceptions or return `None` where an error is appropriate.
- **Explicit over implicit**: Prefer clear, readable code over clever shortcuts. Avoid magic values — use named constants.
- **Pure functions where possible**: Prefer functions without side effects. Side effects (I/O, state mutation) should be isolated and obvious.
- **Immutability by default**: Prefer immutable data structures. Mutate only when there is a clear performance or API reason.
- **Dependency injection**: Pass dependencies in (don't import globals or singletons deep inside logic). Makes testing and swapping implementations easy.
- **Separation of concerns**: Keep I/O (file reads, API calls, DB queries) out of business logic. Business logic should be pure and testable without mocks.

## Error Handling

- Use specific exception types, never bare `except:` or `except Exception:` unless re-raising.
- Always include context in error messages: what failed, what the inputs were, what was expected.
- Log errors at the appropriate level (`warning` for recoverable, `error` for unexpected, `critical` for system-breaking).
- Never hide errors to make the happy path look clean. Propagate or handle — never ignore.

## Testing

- Write tests before or alongside code (TDD preferred for new modules).
- Each test covers one behavior, not one function. Name tests as `test_<scenario>_<expected_outcome>`.
- Tests must be deterministic — no random seeds, no time-dependent logic without injection.
- Avoid mocking internal modules; mock only at system boundaries (external APIs, file system, DB).
- A test that can never fail is worse than no test — assert meaningful postconditions.

## Performance & Scalability

- Don't optimize prematurely — profile before optimizing. Measure, don't guess.
- Avoid N+1 patterns: batch I/O and API calls where possible.
- Be explicit about blocking vs. async operations. Don't mix them carelessly.
- Large inputs should be streamed or paginated — never load unbounded data into memory.

## Maintainability

- Delete dead code — don't comment it out.
- Avoid deeply nested logic (max 2–3 levels). Extract early returns or helper functions to flatten nesting.
- If you copy code more than twice, extract it. If you extract something used once, reconsider.

## Security

- Never log sensitive data (PII, tokens, passwords, medical record contents).
- Treat all external input as untrusted: sanitize, validate, and limit scope.
- Use environment variables for secrets — never hard-code them. Reference `.env.example` for required keys.
- Apply the principle of least privilege: request only the permissions and scopes actually needed.

## Self-Review (run silently before every response containing code)

1. Did I read every file I modified?
2. Is the change minimal — no unasked-for extras?
3. Any security issues or exposed secrets?
4. Does it match requirements in `planning/`?
5. Is the code correct and would it actually run?
6. Are errors handled explicitly with useful messages?
7. Is there a simpler way to achieve the same result?
8. Will the next engineer understand this without asking me?
