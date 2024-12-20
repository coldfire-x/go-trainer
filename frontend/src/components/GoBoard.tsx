import React, { useEffect, useRef, useState } from 'react';
import { Button, Space } from 'antd';
import { UndoOutlined, RedoOutlined } from '@ant-design/icons';

interface GoBoardProps {
  size?: number;
  onMove?: (x: number, y: number, color: 'black' | 'white') => void;
  onUndo?: () => void;
  onRedo?: () => void;
  currentColor?: 'black' | 'white';
  disabled?: boolean;
  canUndo?: boolean;
  canRedo?: boolean;
  boardState: string[][];
  setBoardState: (state: string[][]) => void;
}

export const GoBoard: React.FC<GoBoardProps> = ({
  size = 19,
  onMove,
  onUndo,
  onRedo,
  currentColor = 'black',
  disabled = false,
  canUndo = false,
  canRedo = false,
  boardState,
  setBoardState,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverPosition, setHoverPosition] = useState<[number, number] | null>(null);

  const cellSize = 30;
  const padding = 20;
  const boardWidth = cellSize * (size - 1) + padding * 2;
  const boardHeight = cellSize * (size - 1) + padding * 2;

  useEffect(() => {
    drawBoard();
  }, [boardState, hoverPosition]);

  const drawBoard = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background
    ctx.fillStyle = '#f2b06d';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid lines
    ctx.beginPath();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;

    for (let i = 0; i < size; i++) {
      // Vertical lines
      ctx.moveTo(padding + i * cellSize, padding);
      ctx.lineTo(padding + i * cellSize, boardHeight - padding);
      // Horizontal lines
      ctx.moveTo(padding, padding + i * cellSize);
      ctx.lineTo(boardWidth - padding, padding + i * cellSize);
    }
    ctx.stroke();

    // Draw star points
    const starPoints = size === 19 ? [
      [3, 3], [3, 9], [3, 15],
      [9, 3], [9, 9], [9, 15],
      [15, 3], [15, 9], [15, 15]
    ] : [];

    starPoints.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(
        padding + x * cellSize,
        padding + y * cellSize,
        3,
        0,
        2 * Math.PI
      );
      ctx.fillStyle = '#000000';
      ctx.fill();
    });

    // Draw stones
    boardState.forEach((row, y) => {
      row.forEach((cell, x) => {
        if (cell !== 'empty') {
          drawStone(ctx, x, y, cell);
        }
      });
    });

    // Draw hover stone
    if (hoverPosition && !disabled) {
      const [x, y] = hoverPosition;
      if (boardState[y][x] === 'empty') {
        drawStone(ctx, x, y, currentColor, true);
      }
    }
  };

  const drawStone = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    color: string,
    isHover = false
  ) => {
    const centerX = padding + x * cellSize;
    const centerY = padding + y * cellSize;
    const radius = cellSize * 0.45;

    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    
    if (isHover) {
      ctx.fillStyle = color === 'black' ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)';
    } else {
      // Add gradient effect
      const gradient = ctx.createRadialGradient(
        centerX - radius/3,
        centerY - radius/3,
        radius/10,
        centerX,
        centerY,
        radius
      );

      if (color === 'black') {
        gradient.addColorStop(0, '#666');
        gradient.addColorStop(1, '#000');
      } else {
        gradient.addColorStop(0, '#fff');
        gradient.addColorStop(1, '#ddd');
      }
      ctx.fillStyle = gradient;
    }
    
    ctx.fill();
    ctx.strokeStyle = color === 'black' ? '#000' : '#ccc';
    ctx.lineWidth = 1;
    ctx.stroke();
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert to board coordinates
    const boardX = Math.round((x - padding) / cellSize);
    const boardY = Math.round((y - padding) / cellSize);

    if (
      boardX >= 0 && boardX < size &&
      boardY >= 0 && boardY < size &&
      boardState[boardY][boardX] === 'empty'
    ) {
      const newBoardState = [...boardState];
      newBoardState[boardY][boardX] = currentColor;
      setBoardState(newBoardState);
      onMove?.(boardX, boardY, currentColor);
    }
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const boardX = Math.round((x - padding) / cellSize);
    const boardY = Math.round((y - padding) / cellSize);

    if (
      boardX >= 0 && boardX < size &&
      boardY >= 0 && boardY < size
    ) {
      setHoverPosition([boardX, boardY]);
    } else {
      setHoverPosition(null);
    }
  };

  const handleMouseLeave = () => {
    setHoverPosition(null);
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <canvas
        ref={canvasRef}
        width={boardWidth}
        height={boardHeight}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: disabled ? 'default' : 'pointer' }}
      />
      <div style={{ marginTop: 16 }}>
        <Space>
          <Button
            icon={<UndoOutlined />}
            onClick={onUndo}
            disabled={disabled || !canUndo}
          >
            悔棋
          </Button>
          <Button
            icon={<RedoOutlined />}
            onClick={onRedo}
            disabled={disabled || !canRedo}
          >
            恢复
          </Button>
        </Space>
      </div>
    </div>
  );
};

export default GoBoard;
