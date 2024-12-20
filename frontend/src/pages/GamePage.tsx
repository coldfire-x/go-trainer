import React, { useState } from 'react';
import { Layout, Card, Button, message, Row, Col, Empty, Space, Upload, Switch } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import GoBoard from '../components/GoBoard';
import PositionAnalysis from '../components/PositionAnalysis';
import axios from 'axios';

const { Content } = Layout;

const GamePage: React.FC = () => {
  const [currentColor, setCurrentColor] = useState<'black' | 'white'>('black');
  const [analysis, setAnalysis] = useState<any>(null);
  const [joseki, setJoseki] = useState<string[]>([]);
  const [vitalPoints, setVitalPoints] = useState<[number, number][]>([]);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);
  const [boardState, setBoardState] = useState<string[][]>(
    Array(19).fill(null).map(() => Array(19).fill('empty'))
  );
  const [loading, setLoading] = useState(false);
  const [useML, setUseML] = useState(true);

  const handleNewGame = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/game/new');
      setBoardState(response.data.board);
      setCurrentColor(response.data.current_color);
      setCanUndo(response.data.can_undo);
      setCanRedo(response.data.can_redo);
      setAnalysis(null);
      setGameStarted(true);
      message.success('游戏开始！');
    } catch (error) {
      message.error('创建新游戏失败');
    }
  };

  const handleCreateFromImage = async (file: File) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('use_ml', useML.toString());

      const response = await axios.post('http://localhost:8000/api/game/from_image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('API Response:', response.data);

      const { board: newBoard, current_color, can_undo, can_redo, message } = response.data;
      
      if (!newBoard || !Array.isArray(newBoard)) {
        throw new Error('Invalid board data received');
      }

      setBoardState(newBoard);
      setCurrentColor(current_color);
      setCanUndo(can_undo);
      setCanRedo(can_redo);
      setGameStarted(true);
      message.success(message || 'Game created from image!');
    } catch (error: any) {
      message.error(
        error.response?.data?.detail || 
        error.message || 
        'Failed to create game from image'
      );
      console.error('Error:', error);
      console.error('Response:', error.response?.data);
    } finally {
      setLoading(false);
    }
    return false; // Prevent default upload behavior
  };

  const handleMove = async (x: number, y: number, color: 'black' | 'white') => {
    if (!gameStarted) {
      message.warning('请先开始新游戏');
      return;
    }

    try {
      const moveResponse = await axios.post('http://localhost:8000/api/move', {
        x: x,
        y: y,
        color: color,
      });

      // Update game state
      setBoardState(moveResponse.data.board);
      setCurrentColor(moveResponse.data.current_color);
      setCanUndo(moveResponse.data.can_undo);
      setCanRedo(moveResponse.data.can_redo);

      // Update analysis
      if (moveResponse.data.analysis) {
        setAnalysis(moveResponse.data.analysis);
      }
    } catch (error: any) {
      if (error.response?.data?.detail) {
        message.error(error.response.data.detail);
      } else {
        message.error('下棋失败');
      }
    }
  };

  const handleUndo = async () => {
    if (!gameStarted) {
      message.warning('请先开始新游戏');
      return;
    }

    try {
      const response = await axios.post('http://localhost:8000/api/undo');
      setBoardState(response.data.board);
      setCurrentColor(response.data.current_color);
      setCanUndo(response.data.can_undo);
      setCanRedo(response.data.can_redo);
      setAnalysis(response.data.analysis);
      message.success('已悔棋');
    } catch (error) {
      message.error('悔棋失败');
    }
  };

  const handleRedo = async () => {
    if (!gameStarted) {
      message.warning('请先开始新游戏');
      return;
    }

    try {
      const response = await axios.post('http://localhost:8000/api/redo');
      setBoardState(response.data.board);
      setCurrentColor(response.data.current_color);
      setCanUndo(response.data.can_undo);
      setCanRedo(response.data.can_redo);
      setAnalysis(response.data.analysis);
      message.success('已重做');
    } catch (error) {
      message.error('重做失败');
    }
  };

  return (
    <Layout>
      <Content style={{ padding: '24px' }}>
        <Row gutter={24}>
          <Col span={16}>
            <Card 
              title="围棋对弈" 
              extra={
                <Space>
                  <Button type="primary" onClick={handleNewGame}>
                    {gameStarted ? '重新开始' : '开始游戏'}
                  </Button>
                  <Upload
                    beforeUpload={handleCreateFromImage}
                    showUploadList={false}
                    accept="image/*"
                  >
                    <Button 
                      icon={<UploadOutlined />} 
                      loading={loading}
                    >
                      Create from Image
                    </Button>
                  </Upload>
                  <Space>
                    Use ML:
                    <Switch
                      checked={useML}
                      onChange={setUseML}
                      size="small"
                    />
                  </Space>
                  <Button 
                    onClick={handleUndo} 
                    disabled={!canUndo}
                  >
                    Undo
                  </Button>
                  <Button 
                    onClick={handleRedo} 
                    disabled={!canRedo}
                  >
                    Redo
                  </Button>
                </Space>
              }
            >
              {!gameStarted ? (
                <Empty
                  description="点击 '开始游戏' 按钮开始新的对局"
                  style={{ margin: '40px 0' }}
                />
              ) : (
                <GoBoard
                  onMove={handleMove}
                  onUndo={handleUndo}
                  onRedo={handleRedo}
                  currentColor={currentColor}
                  disabled={!gameStarted}
                  canUndo={canUndo}
                  canRedo={canRedo}
                  boardState={boardState}
                  setBoardState={setBoardState}
                />
              )}
            </Card>
          </Col>
          <Col span={8}>
            <Card title="位置分析">
              <PositionAnalysis
                analysis={analysis}
                joseki={joseki}
                vitalPoints={vitalPoints}
              />
            </Card>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
};

export default GamePage;
