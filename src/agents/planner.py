"""Planner/Recommender Agent - suggests learning paths and next steps"""

from typing import Dict, Any, List, Tuple, Set, Optional
from dataclasses import dataclass, field
import networkx as nx
from loguru import logger
from collections import defaultdict
import random


@dataclass
class LearningItem:
    """Represents a learning recommendation"""
    concept: str
    priority: float
    reasoning: str
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard

    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept': self.concept,
            'priority': self.priority,
            'reasoning': self.reasoning,
            'prerequisites': self.prerequisites,
            'related_concepts': self.related_concepts,
            'difficulty': self.difficulty
        }


@dataclass
class LearningPath:
    """Represents a complete learning path"""
    name: str
    goal: str
    steps: List[LearningItem]
    estimated_time: str = "Unknown"
    total_concepts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'goal': self.goal,
            'steps': [step.to_dict() for step in self.steps],
            'estimated_time': self.estimated_time,
            'total_concepts': self.total_concepts
        }


class PlannerAgent:
    """Agent for planning learning paths and recommendations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = config.get('agents', {}).get('planner', {})

        self.recommendation_count = self.agent_config.get('recommendation_count', 5)
        self.learning_path_depth = self.agent_config.get('learning_path_depth', 3)
        self.personalization_enabled = self.agent_config.get('personalization_enabled', True)

        self.user_profile: Dict[str, Any] = {
            'known_concepts': set(),
            'interests': [],
            'difficulty_preference': 'medium',
            'learning_history': []
        }

        logger.info("Planner Agent initialized")

    def recommend_next_steps(
        self,
        graph_store,
        current_topic: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> List[LearningItem]:
        """Recommend next learning items"""
        logger.info(f"Generating recommendations (current topic: {current_topic})")

        if user_profile:
            self.user_profile.update(user_profile)

        # Get the NetworkX graph
        if hasattr(graph_store, 'store'):
            graph = graph_store.store.graph
        else:
            graph = graph_store.graph

        recommendations = []

        if current_topic:
            # Recommend based on current topic
            recommendations.extend(self._recommend_from_topic(graph, current_topic))
        else:
            # Recommend based on gaps and important concepts
            recommendations.extend(self._recommend_important_concepts(graph))

        # Rank and filter
        recommendations = self._rank_recommendations(recommendations, graph)

        return recommendations[:self.recommendation_count]

    def _recommend_from_topic(self, graph: nx.MultiDiGraph, topic: str) -> List[LearningItem]:
        """Recommend concepts related to a specific topic"""
        recommendations = []

        # Normalize topic name (try to find matching node)
        topic_id = topic.lower().replace(' ', '_')

        if topic_id not in graph.nodes():
            # Try to find partial match
            for node in graph.nodes():
                if topic.lower() in node.lower():
                    topic_id = node
                    break

        if topic_id not in graph.nodes():
            logger.warning(f"Topic '{topic}' not found in graph")
            return recommendations

        # 1. Recommend direct neighbors
        neighbors = list(graph.successors(topic_id))
        for neighbor in neighbors[:5]:
            if neighbor not in self.user_profile['known_concepts']:
                relation = self._get_primary_relation(graph, topic_id, neighbor)
                item = LearningItem(
                    concept=neighbor,
                    priority=0.8,
                    reasoning=f"Directly related to {topic} via {relation}",
                    prerequisites=[topic],
                    difficulty="medium"
                )
                recommendations.append(item)

        # 2. Recommend concepts at distance 2
        for neighbor in neighbors:
            second_order = list(graph.successors(neighbor))
            for concept in second_order[:3]:
                if concept not in self.user_profile['known_concepts'] and concept != topic_id:
                    item = LearningItem(
                        concept=concept,
                        priority=0.6,
                        reasoning=f"Related through {neighbor}",
                        prerequisites=[topic, neighbor],
                        difficulty="medium"
                    )
                    recommendations.append(item)

        # 3. Recommend prerequisites (incoming edges)
        prerequisites = list(graph.predecessors(topic_id))
        for prereq in prerequisites[:3]:
            if prereq not in self.user_profile['known_concepts']:
                item = LearningItem(
                    concept=prereq,
                    priority=0.9,
                    reasoning=f"Prerequisite for understanding {topic}",
                    prerequisites=[],
                    related_concepts=[topic],
                    difficulty="easy"
                )
                recommendations.append(item)

        return recommendations

    def _recommend_important_concepts(self, graph: nx.MultiDiGraph) -> List[LearningItem]:
        """Recommend important concepts based on graph structure"""
        recommendations = []

        if len(graph.nodes()) == 0:
            return recommendations

        try:
            # Use PageRank to find important concepts
            pagerank = nx.pagerank(graph)
            important_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

            for node, score in important_nodes[:10]:
                if node not in self.user_profile['known_concepts']:
                    # Get some context
                    neighbors = list(graph.neighbors(node))[:3]

                    item = LearningItem(
                        concept=node,
                        priority=score,
                        reasoning=f"Central concept in the knowledge graph (importance: {score:.2f})",
                        related_concepts=neighbors,
                        difficulty="medium"
                    )
                    recommendations.append(item)

        except Exception as e:
            logger.warning(f"Could not compute PageRank: {e}")

            # Fallback: use degree centrality
            degrees = dict(graph.degree())
            important_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

            for node, degree in important_nodes[:10]:
                if node not in self.user_profile['known_concepts']:
                    item = LearningItem(
                        concept=node,
                        priority=degree / max(degrees.values()) if degrees.values() else 0,
                        reasoning=f"Well-connected concept ({degree} connections)",
                        difficulty="medium"
                    )
                    recommendations.append(item)

        return recommendations

    def create_learning_path(
        self,
        graph_store,
        goal_concept: str,
        starting_concepts: List[str] = None
    ) -> Optional[LearningPath]:
        """Create a complete learning path to a goal concept"""
        logger.info(f"Creating learning path to '{goal_concept}'")

        # Get the NetworkX graph
        if hasattr(graph_store, 'store'):
            graph = graph_store.store.graph
        else:
            graph = graph_store.graph

        # Normalize goal concept
        goal_id = goal_concept.lower().replace(' ', '_')
        if goal_id not in graph.nodes():
            # Try to find match
            for node in graph.nodes():
                if goal_concept.lower() in node.lower():
                    goal_id = node
                    break

        if goal_id not in graph.nodes():
            logger.warning(f"Goal concept '{goal_concept}' not found in graph")
            return None

        # Find starting point
        if starting_concepts:
            start_nodes = [c.lower().replace(' ', '_') for c in starting_concepts]
            start_nodes = [n for n in start_nodes if n in graph.nodes()]
        else:
            # Find nodes with no incoming edges (fundamentals)
            start_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]

        if not start_nodes:
            start_nodes = list(graph.nodes())[:5]

        # Find best path from any start node to goal
        best_path = None
        shortest_length = float('inf')

        for start in start_nodes:
            try:
                path = nx.shortest_path(graph, start, goal_id)
                if len(path) < shortest_length:
                    best_path = path
                    shortest_length = len(path)
            except nx.NetworkXNoPath:
                continue

        if not best_path:
            logger.warning(f"No path found to '{goal_concept}'")
            return None

        # Build learning path from node path
        steps = []
        for i, node in enumerate(best_path):
            # Determine prerequisites
            prerequisites = best_path[:i] if i > 0 else []

            # Get related concepts
            related = list(graph.neighbors(node))[:3]

            # Estimate difficulty based on position
            if i < len(best_path) / 3:
                difficulty = "easy"
            elif i < 2 * len(best_path) / 3:
                difficulty = "medium"
            else:
                difficulty = "hard"

            # Create learning item
            item = LearningItem(
                concept=node,
                priority=1.0 - (i / len(best_path)),
                reasoning=f"Step {i+1} of {len(best_path)} towards {goal_concept}",
                prerequisites=prerequisites[-2:],  # Last 2 prerequisites
                related_concepts=related,
                difficulty=difficulty
            )
            steps.append(item)

        # Estimate time (rough: 30 min per concept)
        estimated_hours = len(steps) * 0.5
        if estimated_hours < 1:
            time_str = f"{int(estimated_hours * 60)} minutes"
        else:
            time_str = f"{estimated_hours:.1f} hours"

        learning_path = LearningPath(
            name=f"Path to {goal_concept}",
            goal=goal_concept,
            steps=steps,
            estimated_time=time_str,
            total_concepts=len(steps)
        )

        return learning_path

    def identify_weak_areas(self, graph_store, user_profile: Dict[str, Any] = None) -> List[str]:
        """Identify areas where user's knowledge is weak"""
        if user_profile:
            self.user_profile.update(user_profile)

        known = self.user_profile['known_concepts']

        # Get the NetworkX graph
        if hasattr(graph_store, 'store'):
            graph = graph_store.store.graph
        else:
            graph = graph_store.graph

        weak_areas = []

        # Find concepts that are prerequisites to known concepts but not themselves known
        for known_concept in known:
            if known_concept not in graph.nodes():
                continue

            # Get prerequisites (incoming edges)
            prereqs = list(graph.predecessors(known_concept))
            for prereq in prereqs:
                if prereq not in known:
                    weak_areas.append({
                        'concept': prereq,
                        'reasoning': f"Prerequisite for {known_concept} but not yet mastered"
                    })

        return weak_areas[:5]

    def _rank_recommendations(self, recommendations: List[LearningItem], graph: nx.MultiDiGraph) -> List[LearningItem]:
        """Rank recommendations by priority and relevance"""
        # Remove duplicates
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec.concept not in seen:
                seen.add(rec.concept)
                unique_recs.append(rec)

        # Adjust priority based on user profile
        if self.personalization_enabled:
            for rec in unique_recs:
                # Boost if matches interests
                if any(interest.lower() in rec.concept.lower() for interest in self.user_profile['interests']):
                    rec.priority *= 1.2

                # Penalize if too many prerequisites unknown
                unknown_prereqs = [p for p in rec.prerequisites if p not in self.user_profile['known_concepts']]
                if len(unknown_prereqs) > 2:
                    rec.priority *= 0.7

        # Sort by priority
        unique_recs.sort(key=lambda x: x.priority, reverse=True)

        return unique_recs

    def _get_primary_relation(self, graph: nx.MultiDiGraph, source: str, target: str) -> str:
        """Get the primary relation between two nodes"""
        if not graph.has_edge(source, target):
            return "RELATED_TO"

        # Get first relation
        key = list(graph[source][target].keys())[0]
        return graph[source][target][key].get('relation', 'RELATED_TO')

    def update_user_profile(self, learned_concepts: List[str], interests: List[str] = None):
        """Update user profile with learned concepts and interests"""
        self.user_profile['known_concepts'].update(learned_concepts)

        if interests:
            self.user_profile['interests'] = interests

        self.user_profile['learning_history'].extend(learned_concepts)

        logger.info(f"User profile updated: {len(self.user_profile['known_concepts'])} known concepts")

    def get_profile_summary(self) -> str:
        """Get summary of user's learning profile"""
        return f"""Learning Profile:
- Known concepts: {len(self.user_profile['known_concepts'])}
- Interests: {', '.join(self.user_profile['interests']) if self.user_profile['interests'] else 'None specified'}
- Learning history: {len(self.user_profile['learning_history'])} concepts studied
"""
