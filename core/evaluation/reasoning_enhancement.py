"""
Reasoning Enhancement Module for Log Anomaly Detection
Implements self-consistency and verifier feedback techniques to improve SLM performance on AD tasks.
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)


class ReasoningEnhancer:
    """Enhanced reasoning techniques for anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.self_consistency_config = config.get('self_consistency', {})
        self.verifier_config = config.get('verifier_feedback', {})
        
    async def enhance_prediction(self, backend, log_entry: str, dataset_type: str) -> Dict[str, Any]:
        """
        Apply reasoning enhancements to a single log entry prediction
        
        Args:
            backend: The inference backend
            log_entry: The log entry to analyze
            dataset_type: Type of dataset (eventtraces, unsw-nb15)
            
        Returns:
            Enhanced prediction with reasoning details
        """
        # Standard prediction first
        standard_result = await self._standard_prediction(backend, log_entry, dataset_type)
        
        enhanced_result = {
            'log_entry': log_entry,
            'standard_prediction': standard_result,
            'enhancements': {}
        }
        
        # Apply self-consistency if enabled
        if self.self_consistency_config.get('enabled', False):
            logger.info("Applying self-consistency enhancement")
            consistency_result = await self._apply_self_consistency(
                backend, log_entry, dataset_type, standard_result
            )
            enhanced_result['enhancements']['self_consistency'] = consistency_result
            
        # Apply verifier feedback if enabled
        if self.verifier_config.get('enabled', False):
            logger.info("Applying verifier feedback enhancement")
            
            # Use self-consistency result as input if available, otherwise standard
            input_prediction = (
                enhanced_result['enhancements'].get('self_consistency', {}).get('final_prediction')
                or standard_result
            )
            
            verifier_result = await self._apply_verifier_feedback(
                backend, log_entry, dataset_type, input_prediction
            )
            enhanced_result['enhancements']['verifier_feedback'] = verifier_result
            
        # Determine final prediction
        enhanced_result['final_prediction'] = self._get_final_prediction(enhanced_result)
        
        return enhanced_result
    
    async def _standard_prediction(self, backend, log_entry: str, dataset_type: str) -> Dict[str, Any]:
        """Generate standard anomaly detection prediction"""
        
        if dataset_type == "eventtraces":
            prompt = f"""You are a distributed systems expert analyzing log event traces.

Event trace: {log_entry}

Analyze this event trace and determine if it represents an anomaly:

1. Consider the sequence of events
2. Look for unusual patterns or violations of normal system behavior
3. Identify any suspicious event combinations

Respond with:
- "ANOMALY" if the trace shows abnormal behavior
- "NORMAL" if the trace shows normal system behavior
- Provide a brief explanation of your reasoning

Your response:"""
        
        else:  # unsw-nb15
            prompt = f"""You are a network security expert analyzing network traffic data.

Network data: {log_entry}

Analyze this network traffic and determine if it represents an anomaly:

1. Look for suspicious network patterns
2. Check for signs of attacks or malicious activity
3. Consider normal vs abnormal traffic characteristics

Respond with:
- "ANOMALY" if the traffic shows malicious or suspicious behavior
- "NORMAL" if the traffic shows normal network behavior
- Provide a brief explanation of your reasoning

Your response:"""
        
        response = await backend.generate_text(prompt, max_new_tokens=512)
        
        # Parse response
        prediction = "ANOMALY" if "ANOMALY" in response.text.upper() else "NORMAL"
        
        return {
            'prediction': prediction,
            'explanation': response.text,
            'confidence': 0.5,  # Default confidence for standard prediction
            'prompt': prompt
        }
    
    async def _apply_self_consistency(self, backend, log_entry: str, dataset_type: str, 
                                     standard_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply self-consistency reasoning enhancement"""
        
        num_samples = self.self_consistency_config.get('num_samples', 5)
        temperature = self.self_consistency_config.get('temperature', 0.7)
        
        predictions = []
        explanations = []
        prompts = []
        
        for i in range(num_samples):
            # Generate varied prompts for different reasoning paths
            varied_prompt = self._generate_varied_prompt(log_entry, dataset_type, i)
            
            # Generate prediction with higher temperature for diversity
            response = await backend.generate_text(
                varied_prompt, 
                temperature=temperature,
                max_new_tokens=512
            )
            
            # Parse prediction
            prediction = "ANOMALY" if "ANOMALY" in response.text.upper() else "NORMAL"
            
            predictions.append(prediction)
            explanations.append(response.text)
            prompts.append(varied_prompt)
        
        # Calculate consistency metrics
        prediction_counts = Counter(predictions)
        majority_prediction = prediction_counts.most_common(1)[0][0]
        consistency_score = prediction_counts[majority_prediction] / len(predictions)
        
        return {
            'individual_predictions': predictions,
            'individual_explanations': explanations,
            'individual_prompts': prompts,
            'final_prediction': {
                'prediction': majority_prediction,
                'confidence': consistency_score,
                'explanation': f"Majority vote: {majority_prediction} ({prediction_counts[majority_prediction]}/{len(predictions)} samples)"
            },
            'consistency_score': consistency_score,
            'prediction_distribution': dict(prediction_counts)
        }
    
    async def _apply_verifier_feedback(self, backend, log_entry: str, dataset_type: str,
                                      initial_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply verifier feedback reasoning enhancement"""
        
        verifier_temperature = self.verifier_config.get('verifier_temperature', 0.3)
        
        # Generate verifier prompt
        verifier_prompt = self._generate_verifier_prompt(
            log_entry, dataset_type, initial_prediction
        )
        
        # Get verifier response
        verifier_response = await backend.generate_text(
            verifier_prompt, 
            temperature=verifier_temperature,
            max_new_tokens=1024
        )
        
        # Parse verifier decision
        verifier_text = verifier_response.text.upper()
        
        if "REJECT" in verifier_text:
            verification_decision = "REJECT"
        elif "CONFIRM" in verifier_text:
            verification_decision = "CONFIRM"
        else:
            verification_decision = "UNCERTAIN"
        
        # Handle verifier decision
        if verification_decision == "REJECT":
            # Generate refined prediction based on verifier feedback
            refined_prediction = await self._generate_refined_prediction(
                backend, log_entry, dataset_type, initial_prediction, verifier_response.text
            )
            final_prediction = refined_prediction
        else:
            # Enhance confidence based on verification
            confidence_boost = 0.2 if verification_decision == "CONFIRM" else 0.0
            final_prediction = {
                'prediction': initial_prediction['prediction'],
                'confidence': min(1.0, initial_prediction.get('confidence', 0.5) + confidence_boost),
                'explanation': f"Verified: {initial_prediction.get('explanation', '')}"
            }
        
        return {
            'verifier_prompt': verifier_prompt,
            'verifier_response': verifier_response.text,
            'verification_decision': verification_decision,
            'initial_prediction': initial_prediction,
            'final_prediction': final_prediction
        }
    
    def _generate_varied_prompt(self, log_entry: str, dataset_type: str, sample_idx: int) -> str:
        """Generate varied prompts for self-consistency"""
        
        if dataset_type == "eventtraces":
            variations = [
                "You are a distributed systems expert analyzing log event traces.",
                "You are a system administrator investigating potential anomalies in event logs.",
                "You are a cybersecurity analyst examining system event sequences.",
                "You are a software engineer debugging distributed system behavior.",
                "You are a DevOps specialist monitoring system health through logs."
            ]
            
            reasoning_styles = [
                "Think step by step about the event sequence:",
                "Consider the following aspects:",
                "Analyze this systematically:",
                "Examine the trace carefully:",
                "Investigate the following:"
            ]
        else:  # unsw-nb15
            variations = [
                "You are a network security expert analyzing network traffic data.",
                "You are a cybersecurity analyst investigating network anomalies.",
                "You are a network administrator monitoring traffic patterns.",
                "You are a security engineer examining network behavior.",
                "You are a threat hunter analyzing network communications."
            ]
            
            reasoning_styles = [
                "Think step by step about the network traffic:",
                "Consider the following network aspects:",
                "Analyze this traffic systematically:",
                "Examine the network data carefully:",
                "Investigate the following network indicators:"
            ]
        
        role = variations[sample_idx % len(variations)]
        reasoning_style = reasoning_styles[sample_idx % len(reasoning_styles)]
        
        if dataset_type == "eventtraces":
            return f"""{role}

Event trace: {log_entry}

{reasoning_style}
1. What type of event sequence is this?
2. Are there any unusual event patterns?
3. Does this follow normal distributed system behavior?
4. What could indicate an anomaly?

Final classification: NORMAL or ANOMALY
Explanation: Provide your reasoning

Your response:"""
        else:
            return f"""{role}

Network data: {log_entry}

{reasoning_style}
1. What type of network traffic is this?
2. Are there any suspicious network indicators?
3. Does this follow normal network behavior?
4. What could indicate malicious activity?

Final classification: NORMAL or ANOMALY
Explanation: Provide your reasoning

Your response:"""
    
    def _generate_verifier_prompt(self, log_entry: str, dataset_type: str, 
                                 initial_prediction: Dict[str, Any]) -> str:
        """Generate verifier prompt for feedback"""
        
        domain = "distributed systems" if dataset_type == "eventtraces" else "network security"
        
        return f"""You are a senior {domain} expert conducting a critical review of an anomaly detection analysis.

Original Data: {log_entry}
Initial Prediction: {initial_prediction['prediction']}
Initial Reasoning: {initial_prediction.get('explanation', 'No explanation provided')}

As an expert reviewer, critically evaluate this analysis:

1. REASONING QUALITY: Is the logic sound and well-supported?
2. EVIDENCE ANALYSIS: Are there overlooked indicators or misinterpreted signals?
3. ALTERNATIVE PERSPECTIVES: Could there be different valid interpretations?
4. CONFIDENCE ASSESSMENT: How certain should we be about this conclusion?

Critical Questions:
- What evidence supports the opposite conclusion?
- Are there any logical gaps or assumptions?
- What additional context might change the assessment?

Final Verification Decision:
- CONFIRM: The analysis is sound and well-supported
- REJECT: The analysis has significant flaws and should be reconsidered
- UNCERTAIN: The analysis is plausible but confidence is low

Provide your detailed critical assessment and final verification decision:"""
    
    async def _generate_refined_prediction(self, backend, log_entry: str, dataset_type: str,
                                          initial_prediction: Dict[str, Any], 
                                          verifier_feedback: str) -> Dict[str, Any]:
        """Generate refined prediction based on verifier feedback"""
        
        refinement_prompt = f"""Based on the critical feedback below, provide a refined analysis of this data:

Original Data: {log_entry}
Initial Prediction: {initial_prediction['prediction']}
Initial Reasoning: {initial_prediction.get('explanation', '')}

Critical Feedback:
{verifier_feedback}

Considering the feedback, provide a refined analysis:

1. Address the specific concerns raised in the feedback
2. Incorporate any overlooked evidence or alternative perspectives
3. Provide a more robust conclusion

Refined Classification: NORMAL or ANOMALY
Improved Explanation: Your enhanced reasoning

Your refined response:"""
        
        refined_response = await backend.generate_text(refinement_prompt, max_new_tokens=1024)
        
        # Parse refined response
        refined_prediction = "ANOMALY" if "ANOMALY" in refined_response.text.upper() else "NORMAL"
        
        return {
            'prediction': refined_prediction,
            'confidence': 0.7,  # Higher confidence for refined prediction
            'explanation': refined_response.text,
            'refinement_prompt': refinement_prompt
        }
    
    def _get_final_prediction(self, enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the final prediction from all enhancements"""
        
        # Priority order: 
        # ... verifier_feedback > self_consistency > standard
        if 'verifier_feedback' in enhanced_result['enhancements']:
            return enhanced_result['enhancements']['verifier_feedback']['final_prediction']
        elif 'self_consistency' in enhanced_result['enhancements']:
            return enhanced_result['enhancements']['self_consistency']['final_prediction']
        else:
            return enhanced_result['standard_prediction']