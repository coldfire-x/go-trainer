import React, { useState } from 'react';
import { Layout, Card, Button, message, Row, Col, Empty, Space, Upload, Image } from 'antd';
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
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const handleNewGame = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/game/new');
      setBoardState(response.data.board);
      setCurrentColor(response.data.current_color);
      setCanUndo(response.data.can_undo);
      setCanRedo(response.data.can_redo);
      setAnalysis(null);
      setGameStarted(true);
      setImageUrl(null); // Clear image when starting new game
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

      // Create object URL for the uploaded image
      const objectUrl = URL.createObjectURL(file);
      setImageUrl(objectUrl);

      const response = await axios.post('http://localhost:8000/api/game/from_image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('API Response:', response.data);

      const { board: newBoard, current_color, can_undo, can_redo, message: responseMessage } = response.data;
      
      if (!newBoard || !Array.isArray(newBoard)) {
        throw new Error('Invalid board data received');
      }

      setBoardState(newBoard);
      setCurrentColor(current_color);
      setCanUndo(can_undo);
      setCanRedo(can_redo);
      setGameStarted(true);
      message.success(responseMessage || 'Game created from image!');
    } catch (error: any) {
      message.error(
        error.response?.data?.detail || 
        error.message || 
        'Failed to create game from image'
      );
      console.error('Error:', error);
      console.error('Response:', error.response?.data);
      setImageUrl(null); // Clear image on error
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

      const {
        board: newBoard,
        current_color,
        can_undo,
        can_redo,
        analysis: newAnalysis,
      } = moveResponse.data;

      setBoardState(newBoard);
      setCurrentColor(current_color);
      setCanUndo(can_undo);
      setCanRedo(can_redo);
      setAnalysis(newAnalysis);

      if (newAnalysis?.joseki) {
        setJoseki(newAnalysis.joseki);
      }
      if (newAnalysis?.vital_points) {
        setVitalPoints(newAnalysis.vital_points);
      }
    } catch (error: any) {
      message.error(error.response?.data?.detail || '落子失败');
    }
  };

  const handleUndo = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/undo');
      setBoardState(response.data.board);
      setCurrentColor(response.data.current_color);
      setCanUndo(response.data.can_undo);
      setCanRedo(response.data.can_redo);
      setAnalysis(response.data.analysis);
    } catch (error: any) {
      message.error(error.response?.data?.detail || '撤销失败');
    }
  };

  const handleRedo = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/redo');
      setBoardState(response.data.board);
      setCurrentColor(response.data.current_color);
      setCanUndo(response.data.can_undo);
      setCanRedo(response.data.can_redo);
      setAnalysis(response.data.analysis);
    } catch (error: any) {
      message.error(error.response?.data?.detail || '重做失败');
    }
  };

  return (
    <Layout>
      <Content style={{ padding: '24px' }}>
        <Row gutter={24}>
          <Col span={16}>
            <Card
              title="围棋对局"
              loading={loading}
              extra={
                <Space>
                  <Button onClick={handleNewGame} type="primary">
                    开始游戏
                  </Button>
                  <Upload
                    showUploadList={false}
                    beforeUpload={handleCreateFromImage}
                    accept="image/*"
                  >
                    <Button icon={<UploadOutlined />} loading={loading}>
                      从图片创建
                    </Button>
                  </Upload>
                </Space>
              }
            >
              {!gameStarted ? (
                <Empty
                  description={
                    <Space direction="vertical" align="center">
                      <div>点击 '开始游戏' 开始新的对局</div>
                      <div>或上传棋盘图片继续对局</div>
                      <div style={{ fontSize: '12px', color: '#999' }}>
                        提示: 使用右键菜单或键盘快捷键 (Z/Y) 进行悔棋/重做
                      </div>
                    </Space>
                  }
                  style={{ margin: '40px 0' }}
                />
              ) : (
                <div>
                  <div style={{ marginBottom: '8px', fontSize: '12px', color: '#666' }}>
                    快捷键: Z - 悔棋, Y - 重做
                  </div>
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
                </div>
              )}
            </Card>
          </Col>
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              {imageUrl && (
                <Card 
                  size="small"
                  title={
                    <div style={{ fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span>原始图片</span>
                      <a 
                        href={imageUrl} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={{ fontSize: '12px' }}
                      >
                        查看原图
                      </a>
                    </div>
                  }
                  bodyStyle={{ padding: '8px' }}
                >
                  <div style={{ 
                    width: '100%', 
                    height: '200px', 
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    overflow: 'hidden',
                    background: '#f5f5f5',
                    borderRadius: '4px'
                  }}>
                    <Image
                      src={imageUrl}
                      alt="Original board"
                      style={{ 
                        maxWidth: '100%',
                        maxHeight: '200px',
                        objectFit: 'contain'
                      }}
                      preview={{
                        mask: '点击查看大图'
                      }}
                    />
                  </div>
                </Card>
              )}
              <Card title="位置分析">
                <PositionAnalysis
                  analysis={analysis}
                  joseki={joseki}
                  vitalPoints={vitalPoints}
                />
              </Card>
            </Space>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
};

export default GamePage;
