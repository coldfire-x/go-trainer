import React, { useState } from 'react';
import { Layout, Card, Upload, Button, message, Row, Col } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import GoBoard from '../components/GoBoard';
import PositionAnalysis from '../components/PositionAnalysis';
import axios from 'axios';

const { Content } = Layout;

const GamePage: React.FC = () => {
  const [currentColor, setCurrentColor] = useState<'black' | 'white'>('black');
  const [gameId, setGameId] = useState<number | null>(null);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [analysis, setAnalysis] = useState<any>(null);
  const [joseki, setJoseki] = useState<string[]>([]);
  const [vitalPoints, setVitalPoints] = useState<[number, number][]>([]);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [boardState, setBoardState] = useState<string[][]>(
    Array(19).fill(null).map(() => Array(19).fill('empty'))
  );

  const updateBoardState = async () => {
    if (!gameId) return;
    
    try {
      const response = await axios.get(`http://localhost:8000/api/games/${gameId}/state`);
      setBoardState(response.data.board);
      setCurrentColor(response.data.current_color);
      setCanUndo(response.data.can_undo);
      setCanRedo(response.data.can_redo);
    } catch (error) {
      message.error('获取棋盘状态失败');
    }
  };

  const handleMove = async (x: number, y: number, color: 'black' | 'white') => {
    try {
      if (!gameId) {
        const response = await axios.post('http://localhost:8000/api/games/new');
        setGameId(response.data.game_id);
        setCurrentColor(response.data.current_color);
      }

      const moveResponse = await axios.post(`http://localhost:8000/api/games/${gameId}/move`, {
        x,
        y,
        color,
      });

      if (moveResponse.data.position_analysis) {
        setAnalysis(moveResponse.data.position_analysis);
      }
      if (moveResponse.data.joseki) {
        setJoseki(moveResponse.data.joseki);
      }
      setCanUndo(moveResponse.data.can_undo);
      setCanRedo(moveResponse.data.can_redo);
      setCurrentColor(moveResponse.data.current_color);

      // Get position analysis
      const analysisResponse = await axios.get(
        `http://localhost:8000/api/games/${gameId}/analysis`,
        { params: { x, y } }
      );
      if (analysisResponse.data.vital_points) {
        setVitalPoints(analysisResponse.data.vital_points);
      }

      // Update board state
      await updateBoardState();
    } catch (error: any) {
      if (error.response?.data?.detail) {
        message.error(error.response.data.detail);
      } else {
        message.error('下棋失败');
      }
    }
  };

  const handleUndo = async () => {
    try {
      if (gameId) {
        const response = await axios.post(`http://localhost:8000/api/games/${gameId}/undo`);
        setAnalysis(null);
        setJoseki([]);
        setVitalPoints([]);
        setCanUndo(response.data.can_undo);
        setCanRedo(response.data.can_redo);
        setCurrentColor(response.data.current_color);

        // Update board state
        await updateBoardState();

        message.success('已悔棋');
      }
    } catch (error) {
      message.error('悔棋失败');
    }
  };

  const handleRedo = async () => {
    try {
      if (gameId) {
        const response = await axios.post(`http://localhost:8000/api/games/${gameId}/redo`);
        
        // Get position analysis for the redone move
        if (response.data.next_move) {
          const [x, y] = response.data.next_move;
          const analysisResponse = await axios.get(
            `http://localhost:8000/api/games/${gameId}/analysis`,
            { params: { x, y } }
          );
          if (analysisResponse.data.analysis) {
            setAnalysis(analysisResponse.data.analysis);
          }
          if (analysisResponse.data.vital_points) {
            setVitalPoints(analysisResponse.data.vital_points);
          }
        }
        
        setCanUndo(response.data.can_undo);
        setCanRedo(response.data.can_redo);
        setCurrentColor(response.data.current_color);

        // Update board state
        await updateBoardState();

        message.success('已恢复');
      }
    } catch (error) {
      message.error('恢复失败');
    }
  };

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post(
        'http://localhost:8000/api/recognition/board',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      if (response.data.board_state) {
        setBoardState(response.data.board_state);
        message.success('棋盘识别成功');
      }
    } catch (error) {
      message.error('棋盘识别失败');
    }

    return false;
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{ padding: '50px' }}>
        <Row gutter={16}>
          <Col span={16}>
            <Card
              title="围棋练习"
              extra={
                <Upload
                  fileList={fileList}
                  beforeUpload={handleUpload}
                  onChange={({ fileList }) => setFileList(fileList)}
                >
                  <Button icon={<UploadOutlined />}>上传题目</Button>
                </Upload>
              }
            >
              <GoBoard
                onMove={handleMove}
                onUndo={handleUndo}
                onRedo={handleRedo}
                currentColor={currentColor}
                canUndo={canUndo}
                canRedo={canRedo}
                boardState={boardState}
                setBoardState={setBoardState}
              />
            </Card>
          </Col>
          <Col span={8}>
            <PositionAnalysis
              analysis={analysis}
              joseki={joseki}
              vitalPoints={vitalPoints}
            />
          </Col>
        </Row>
      </Content>
    </Layout>
  );
};

export default GamePage;
