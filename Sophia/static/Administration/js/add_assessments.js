function validateInput(event) {
    const inputValue = event.target.value;

    // Remove spaces and replace with underscores
    const sanitizedValue = inputValue.replace(/\s+/g, '_');

    // Check for characters that are not allowed (non-alphabetic, non-underscore, non-numeric)
    const regex = /[^A-Za-z0-9_]/g;
    if (regex.test(sanitizedValue)) {
      const validCharacters = sanitizedValue.replace(regex, '');
      event.target.value = validCharacters;
    } else {
      event.target.value = sanitizedValue;
    }
  }