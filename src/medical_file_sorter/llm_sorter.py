"""
LLMSorter class for classifying and grouping medical documents using multimodal LLMs.
Supports multiple backends: OpenAI, OpenRouter (cloud), and LM Studio (local).
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from openai import OpenAI

# System prompt for the medical document sorting task (text-based, Stage 2)
SYSTEM_PROMPT = """You are a medical administrative assistant specialized in organizing records.

You will receive EXTRACTED TEXT CONTENT from medical documents. Your task is to:

1. **Classify** each document as either:
   - "Prescription": Documents containing "Rx", doctor signatures, medical advice, or medication instructions
   - "Bill": Documents containing currency symbols (₹, $), "Total", "GST", "Invoice", payment details, or itemized charges

2. **Group** related documents into transactions based on:
   - **Content matching is the priority**: If a prescription lists specific medicines (e.g., "Paracetamol", "Amoxicillin") and a bill lists the same medicines, they belong together
   - Doctor names appearing on both prescription and related bills
   - Hospital/clinic names matching between documents
   - Dates can be a supporting factor but are NOT sufficient alone for matching

3. **Create transactions** where each group contains:
   - 1 Prescription + N related Bills, OR
   - N Prescriptions + 1 related Bill
   - Sort each group with Prescription(s) first, then Bill(s)

4. **Extract key information** for each group:
   - **Patient name**: CRITICAL - Look carefully for the patient's name:
     * **PRIORITY 1**: If a field is explicitly labeled "Patient:", "Pt Name:", "Patient Name:", "Customer Name:", "Bill To:" - that is ALWAYS the patient's name, even if they have a "Dr." title
     * **PRIORITY 2**: On prescriptions, check the header area where patient details (name, age, gender) are written together
     * **How to identify the PRESCRIBING DOCTOR (not the patient)**: The prescribing/treating doctor's name typically has MEDICAL DEGREES after it (MBBS, MD, MS, DNB, FRCS, etc.) and appears in the letterhead or signature area
     * **Referring Doctor**: Names labeled "Ref. by", "Referred by", "Consultant" are referring doctors, not patients
     * A patient CAN have a "Dr." title if they are a doctor themselves - don't exclude names just because of "Dr." prefix
   - **Bill amount**: Sum up the total bill amount from all bills in the group (numeric value only, no currency symbol). Look for "Total", "Grand Total", "Amount Payable", "Net Amount"
   - **Summary**: Brief explanation of what documents contain and why they're grouped

5. **Handle unmatched documents**: Any document that cannot be confidently matched should go in "uncategorized"

**IMPORTANT**: 
- Return ONLY valid JSON with no markdown formatting or code fences
- Use exact filenames as provided
- Be conservative - if unsure about a match, put documents in uncategorized
- For bill_amount, use 0 if no bill is present or amount cannot be determined
- For patient_name, try hard to find the name - only use "Unknown" if there is genuinely no patient name visible on any document in the group

Return your response in this exact JSON format:
{
    "groups": [
        {
            "files": ["prescription1.pdf", "bill1.pdf", "bill2.jpg"],
            "patient_name": "John Doe",
            "bill_amount": 1250.50,
            "summary": "Prescription and bills from Dr. Smith at City Hospital for antibiotics (Amoxicillin)"
        },
        {
            "files": ["prescription2.png", "bill3.pdf"],
            "patient_name": "Jane Smith",
            "bill_amount": 500.00,
            "summary": "Eye checkup prescription and pharmacy bill from Vision Care Clinic"
        }
    ],
    "uncategorized": ["unknown1.pdf", "unclear_document.jpg"]
}"""


class LLMBackend(Enum):
    """Supported LLM backends."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LM_STUDIO = "lmstudio"


@dataclass
class BackendConfig:
    """Configuration for an LLM backend."""
    name: str
    base_url: Optional[str]
    default_model: str
    requires_api_key: bool
    description: str


# Backend configurations
BACKEND_CONFIGS: dict[LLMBackend, BackendConfig] = {
    LLMBackend.OPENAI: BackendConfig(
        name="OpenAI",
        base_url=None,  # Uses default OpenAI URL
        default_model="gpt-4o",
        requires_api_key=True,
        description="OpenAI API (cloud)"
    ),
    LLMBackend.OPENROUTER: BackendConfig(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o",
        requires_api_key=True,
        description="OpenRouter API (cloud, multi-provider)"
    ),
    LLMBackend.LM_STUDIO: BackendConfig(
        name="LM Studio",
        base_url="http://localhost:1234/v1",
        default_model="openai/gpt-oss-20b",
        requires_api_key=False,
        description="LM Studio (local inference)"
    ),
}


class LLMSorter:
    """
    Uses multimodal LLMs to classify and group medical documents.
    Supports OpenAI, OpenRouter, and LM Studio backends.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        backend: LLMBackend = LLMBackend.OPENROUTER,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLMSorter.

        Args:
            api_key: API key (not required for LM Studio).
            model: Model to use. If None, uses the backend's default.
            backend: Which backend to use (openai, openrouter, lmstudio).
            base_url: Custom base URL (overrides backend default).
        """
        self.backend = backend
        self.config = BACKEND_CONFIGS[backend]
        
        # Determine model
        self.model = model or self.config.default_model
        
        # Determine base URL
        effective_base_url = base_url or self.config.base_url
        
        # Validate API key requirement
        if self.config.requires_api_key and not api_key:
            raise ValueError(f"{self.config.name} requires an API key")
        
        # Initialize the OpenAI client (works with all OpenAI-compatible APIs)
        client_kwargs = {}
        if effective_base_url:
            client_kwargs["base_url"] = effective_base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            # LM Studio doesn't need a real key, but the client requires one
            client_kwargs["api_key"] = "lm-studio"
        
        self.client = OpenAI(**client_kwargs)
        
        # Additional headers for OpenRouter
        self._extra_headers = {}
        if backend == LLMBackend.OPENROUTER:
            self._extra_headers = {
                "HTTP-Referer": "https://github.com/medical-file-sorter",
                "X-Title": "Medical File Sorter"
            }

    def _create_image_content(
        self, filename: str, base64_data: str
    ) -> list[dict[str, Any]]:
        """
        Create the content array for a single image.

        Args:
            filename: Name of the file.
            base64_data: Base64 encoded image data.

        Returns:
            List of content dictionaries for the API.
        """
        return [
            {"type": "text", "text": f"File: {filename}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_data}",
                    "detail": "low",
                },
            },
        ]

    def sort_documents(self, file_data: dict[str, str]) -> dict[str, Any]:
        """
        Send documents to the LLM for classification and grouping.

        Args:
            file_data: Dictionary mapping filenames to Base64 encoded strings.

        Returns:
            Dictionary with 'groups' and 'uncategorized' keys.

        Raises:
            ValueError: If API returns invalid response.
            Exception: If API call fails.
        """
        if not file_data:
            return {"groups": [], "uncategorized": []}

        # Build the user message with all images
        user_content = [
            {
                "type": "text",
                "text": f"Please analyze and organize these {len(file_data)} medical documents:",
            }
        ]

        for filename, base64_data in file_data.items():
            user_content.extend(self._create_image_content(filename, base64_data))

        try:
            # Build messages - some models (like Gemma) don't support system prompts
            # For those, prepend system prompt to user message
            model_lower = self.model.lower()
            if "gemma" in model_lower:
                # Gemma doesn't support system prompts, prepend to user content
                user_content.insert(0, {"type": "text", "text": SYSTEM_PROMPT + "\n\n"})
                messages = [{"role": "user", "content": user_content}]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

            # Build request kwargs
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.1,  # Low temperature for consistent results
            }
            
            # Add extra headers for OpenRouter
            if self._extra_headers:
                request_kwargs["extra_headers"] = self._extra_headers

            response = self.client.chat.completions.create(**request_kwargs)

            # Check for empty response
            if not response.choices:
                raise ValueError("LLM returned no choices - model may not support vision")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content - model may not support vision or the request")
            
            result_text = content.strip()

            # Try to parse the JSON response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()

                result = json.loads(result_text)

            # Validate the response structure
            if "groups" not in result:
                result["groups"] = []
            if "uncategorized" not in result:
                result["uncategorized"] = []

            return result

        except Exception as e:
            raise Exception(f"Failed to sort documents with {self.config.name}: {e}")

    def sort_documents_by_text(self, extracted_content: dict[str, str]) -> dict[str, Any]:
        """
        Sort documents using extracted text content (Stage 2 of pipeline).

        Args:
            extracted_content: Dictionary mapping filenames to extracted text content.

        Returns:
            Dictionary with 'groups' and 'uncategorized' keys.

        Raises:
            ValueError: If API returns invalid response.
            Exception: If API call fails.
        """
        if not extracted_content:
            return {"groups": [], "uncategorized": []}

        # Build the user message with all text content
        user_message = f"Please analyze and organize these {len(extracted_content)} medical documents based on their extracted text content:\n\n"
        
        for filename, text_content in extracted_content.items():
            user_message += f"=== FILE: {filename} ===\n"
            # Truncate very long content to avoid token limits
            if len(text_content) > 5000:
                user_message += text_content[:5000] + "\n[... content truncated ...]\n"
            else:
                user_message += text_content + "\n"
            user_message += "\n"

        try:
            # Build messages
            model_lower = self.model.lower()
            if "gemma" in model_lower:
                messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_message}]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]

            # Build request kwargs
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.1,
            }
            
            if self._extra_headers:
                request_kwargs["extra_headers"] = self._extra_headers

            response = self.client.chat.completions.create(**request_kwargs)

            if not response.choices:
                raise ValueError("LLM returned no choices")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content")
            
            result_text = content.strip()

            # Try to parse the JSON response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()

                result = json.loads(result_text)

            # Validate the response structure
            if "groups" not in result:
                result["groups"] = []
            if "uncategorized" not in result:
                result["uncategorized"] = []

            return result

        except Exception as e:
            raise Exception(f"Failed to sort documents with {self.config.name}: {e}")

    def validate_result(
        self, result: dict[str, Any], known_files: set[str]
    ) -> dict[str, Any]:
        """
        Validate and clean the LLM result, removing any hallucinated filenames.

        Args:
            result: The result from sort_documents.
            known_files: Set of actual filenames in the folder.

        Returns:
            Cleaned result with only valid filenames.
        """
        validated_groups = []

        for group in result.get("groups", []):
            # Handle both old format (list) and new format (dict with files/summary)
            if isinstance(group, dict):
                files = group.get("files", [])
                summary = group.get("summary", "")
                patient_name = group.get("patient_name", "Unknown")
                bill_amount = group.get("bill_amount", 0)
            else:
                # Legacy format: group is just a list of files
                files = group
                summary = ""
                patient_name = "Unknown"
                bill_amount = 0

            valid_files = []
            for filename in files:
                if filename in known_files:
                    valid_files.append(filename)
                else:
                    print(f"Warning: LLM returned unknown filename: {filename}")
            
            if valid_files:
                validated_groups.append({
                    "files": valid_files,
                    "summary": summary,
                    "patient_name": patient_name,
                    "bill_amount": float(bill_amount) if bill_amount else 0.0
                })

        validated_uncategorized = []
        for filename in result.get("uncategorized", []):
            if filename in known_files:
                validated_uncategorized.append(filename)
            else:
                print(f"Warning: LLM returned unknown filename: {filename}")

        # Find any files that weren't mentioned at all
        mentioned_files = set()
        for group in validated_groups:
            mentioned_files.update(group["files"])
        mentioned_files.update(validated_uncategorized)

        missing_files = known_files - mentioned_files
        if missing_files:
            print(f"Warning: LLM did not mention these files: {missing_files}")
            validated_uncategorized.extend(sorted(missing_files))

        return {"groups": validated_groups, "uncategorized": validated_uncategorized}


def get_backend_from_string(backend_str: str) -> LLMBackend:
    """
    Convert a string to LLMBackend enum.
    
    Args:
        backend_str: Backend name (openai, openrouter, lmstudio).
        
    Returns:
        LLMBackend enum value.
        
    Raises:
        ValueError: If invalid backend name.
    """
    backend_map = {
        "openai": LLMBackend.OPENAI,
        "openrouter": LLMBackend.OPENROUTER,
        "lmstudio": LLMBackend.LM_STUDIO,
        "lm-studio": LLMBackend.LM_STUDIO,
        "local": LLMBackend.LM_STUDIO,
    }
    
    normalized = backend_str.lower().strip()
    if normalized not in backend_map:
        valid = ", ".join(backend_map.keys())
        raise ValueError(f"Invalid backend '{backend_str}'. Valid options: {valid}")
    
    return backend_map[normalized]


def list_backends() -> str:
    """
    Get a formatted string listing all available backends.
    
    Returns:
        Formatted string with backend information.
    """
    lines = ["Available backends:"]
    for backend, config in BACKEND_CONFIGS.items():
        lines.append(f"  {backend.value:12} - {config.description} (default model: {config.default_model})")
    return "\n".join(lines)
