import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict
import os

class TradingVisualizer:
    """Visualization tools for trading performance and training metrics."""
    
    def __init__(self, style='seaborn'):
        """Initialize visualizer with plotting style."""
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_portfolio_performance(self, portfolio_values: List[float], 
                                 trade_history: List[tuple] = None,
                                 save_path: str = None) -> None:
        """Plot portfolio value over time with trade markers."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value
        ax1.plot(portfolio_values, linewidth=2, label='Portfolio Value')
        ax1.set_title('Portfolio Performance Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mark trades if provided
        if trade_history:
            buy_points = []
            sell_points = []
            for trade in trade_history:
                action, shares, price, step = trade
                if step < len(portfolio_values):
                    if action == 'BUY':
                        buy_points.append((step, portfolio_values[step]))
                    elif action == 'SELL':
                        sell_points.append((step, portfolio_values[step]))
            
            if buy_points:
                buy_x, buy_y = zip(*buy_points)
                ax1.scatter(buy_x, buy_y, color='green', marker='^', s=50, label='Buy', alpha=0.7)
            if sell_points:
                sell_x, sell_y = zip(*sell_points)
                ax1.scatter(sell_x, sell_y, color='red', marker='v', s=50, label='Sell', alpha=0.7)
            ax1.legend()
        
        # Returns
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        ax2.plot(returns, linewidth=1, alpha=0.7, color='orange')
        ax2.set_title('Daily Returns')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_metrics(self, losses: List[float], 
                            q_values: List[float],
                            epsilon_values: List[float] = None,
                            save_path: str = None) -> None:
        """Plot training metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        if losses:
            axes[0, 0].plot(losses, alpha=0.7)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Moving average
            if len(losses) > 50:
                ma_losses = pd.Series(losses).rolling(50).mean()
                axes[0, 0].plot(ma_losses, color='red', linewidth=2, label='MA(50)')
                axes[0, 0].legend()
        
        # Q-values
        if q_values:
            axes[0, 1].plot(q_values, alpha=0.7, color='green')
            axes[0, 1].set_title('Average Q-Values')
            axes[0, 1].set_ylabel('Q-Value')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Moving average
            if len(q_values) > 50:
                ma_q = pd.Series(q_values).rolling(50).mean()
                axes[0, 1].plot(ma_q, color='red', linewidth=2, label='MA(50)')
                axes[0, 1].legend()
        
        # Epsilon decay
        if epsilon_values:
            axes[1, 0].plot(epsilon_values, color='purple')
            axes[1, 0].set_title('Epsilon Decay')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss distribution
        if losses:
            axes[1, 1].hist(losses, bins=50, alpha=0.7, color='orange')
            axes[1, 1].set_title('Loss Distribution')
            axes[1, 1].set_xlabel('Loss')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, float],
                              benchmark_metrics: Dict[str, float] = None,
                              save_path: str = None) -> None:
        """Plot performance metrics comparison."""
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(metrics_names))
        bars = ax.bar(x_pos, metrics_values, alpha=0.7, color='skyblue', label='Agent')
        
        # Add benchmark if provided
        if benchmark_metrics:
            benchmark_values = [benchmark_metrics.get(name, 0) for name in metrics_names]
            ax.bar(x_pos + 0.35, benchmark_values, width=0.35, alpha=0.7, 
                  color='lightcoral', label='Benchmark')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Trading Performance Metrics')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_portfolio(self, portfolio_values: List[float],
                                 trade_history: List[tuple] = None) -> go.Figure:
        """Create interactive portfolio performance plot using Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value', 'Daily Returns'),
            vertical_spacing=0.1
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Add trade markers
        if trade_history:
            buy_steps, buy_values = [], []
            sell_steps, sell_values = [], []
            
            for trade in trade_history:
                action, shares, price, step = trade
                if step < len(portfolio_values):
                    if action == 'BUY':
                        buy_steps.append(step)
                        buy_values.append(portfolio_values[step])
                    elif action == 'SELL':
                        sell_steps.append(step)
                        sell_values.append(portfolio_values[step])
            
            if buy_steps:
                fig.add_trace(
                    go.Scatter(
                        x=buy_steps, y=buy_values,
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=8, color='green')
                    ),
                    row=1, col=1
                )
            
            if sell_steps:
                fig.add_trace(
                    go.Scatter(
                        x=sell_steps, y=sell_values,
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=8, color='red')
                    ),
                    row=1, col=1
                )
        
        # Returns
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        fig.add_trace(
            go.Scatter(
                y=returns,
                mode='lines',
                name='Returns',
                line=dict(width=1, color='orange')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Interactive Trading Performance Dashboard',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time Steps", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Returns", row=2, col=1)
        
        return fig
