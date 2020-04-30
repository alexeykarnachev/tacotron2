# Changelog
All notable changes to this project will be documented in this file.

## [0.0.3] - 2020-04-30
### Fixed
- Telegram bot behavior in exception case
- Telegram bot refactored

### Added
- Default denoising strength for payload without this parameter in application
- User-readable exception in case of incorrect text in telegram bor
- Waveglow denoiser modet to tacotron2 package and now is available for other vocoders
- Vocoder module and interfaces

### Updated
- torch 1.3.1 -> 1.4.0


## [0.0.2] - 2020-04-17
### Fixed
- Bot behavior before /start is sent.
- Phonemizer refactoring (exceptions, saving already encoded words to object state).
- 'name' of Speak view fixed for Swagger.
- Default voice issue fixed in telegram bot.
- tokenizers now has version 0.5.0

### Added
- Option to set denoising strength in app request.
- Option to use phonemes instead of text in BaseEvaluator.
- Full pytorch-lightning enviroment: module, train sctript, usage of lightning-checkpoint in evaluators.
- `+` sign as an accent identificator through evaluator, application, telegram bot.
- Now bot send not mp3-file, but voice message.
- Healthcheck and version added for application.
- Tests: evaluator, preprocessors, application.